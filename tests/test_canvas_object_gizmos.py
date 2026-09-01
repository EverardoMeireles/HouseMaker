# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import trimesh
from trimesh.visual.texture import TextureVisuals

from housemaker.glb import GeneratedModel, PreviewPlacedObject
from housemaker.glass_material import build_housemaker_glass_material
from housemaker.surface_geometry import FixedSurface, SURFACE_TYPE_WALL
from housemaker.viewer import (
    NAVIGATION_MODE_FIRST_PERSON,
    GlbViewerWidget,
    _TransformGizmoHandle,
    _build_axis_drag_plane_normal,
    _get_nearest_preview_placed_object_ray_hit,
    _get_signed_rotation_degrees,
    _intersect_ray_with_plane,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test doubles ###
class _MouseButtonEvent:
    def __init__(
        self,
        *,
        button: Qt.MouseButton,
        position: QPointF,
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
def _translation_transform(
    x: float,
    y: float,
    z: float,
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (x, y, z)
    return transform


def _build_local_box(
    *,
    x_center: float = 0.0,
    x_extent: float = 1.0,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(x_extent, 1.0, 1.0))
    mesh.apply_translation((x_center, 0.0, 0.5))
    return mesh


def _build_placed_object(
    object_id: str,
    *,
    world_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0),
    symmetric: bool = False,
) -> PreviewPlacedObject:
    local_mesh = (
        _build_local_box(x_center=0.4, x_extent=0.8)
        if symmetric
        else _build_local_box()
    )
    return PreviewPlacedObject(
        object_id=object_id,
        meshes=(local_mesh,),
        placement_transform=_translation_transform(*world_position),
        world_position=world_position,
        rotation_degrees=rotation_degrees,
        symmetric_preview_orientation="vertical" if symmetric else None,
        symmetric_preview_plane_coordinate=0.0 if symmetric else None,
    )


def _build_preview_model(
    *placed_objects: PreviewPlacedObject,
) -> GeneratedModel:
    base_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    base_mesh.apply_translation((100.0, 100.0, 100.0))
    return GeneratedModel(
        mesh=base_mesh,
        scene=trimesh.Scene(base_mesh),
        glb_bytes=b"",
        preview_placed_objects=list(placed_objects),
        preview_base_mesh=base_mesh.copy(),
    )


def _build_wall(
    *,
    surface_id: str = "level:0/wall:1:2",
    y: float = 2.0,
) -> FixedSurface:
    vertices = np.asarray(
        (
            (-2.0, y, 0.0),
            (2.0, y, 0.0),
            (2.0, y, 2.0),
            (-2.0, y, 2.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=8.0,
        wall_key="1:2",
        wall_start_world=(-2.0, y, 0.0),
        wall_end_world=(2.0, y, 0.0),
        wall_height_meters=2.0,
    )


def _forward_ray(
    *,
    x: float = 0.0,
    y: float = -5.0,
    z: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((x, y, z), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


def _downward_ray(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((x, y, 5.0), dtype=float),
        np.asarray((0.0, 0.0, -1.0), dtype=float),
    )


# ### Geometry helper tests ###
class TransformGizmoGeometryTests(unittest.TestCase):
    def test_ray_plane_intersection_rejects_parallel_and_behind_hits(self) -> None:
        hit = _intersect_ray_with_plane(
            np.asarray((2.0, -3.0, 4.0), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
            np.asarray((0.0, 2.0, 0.0), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )

        np.testing.assert_allclose(hit, (2.0, 2.0, 4.0))
        self.assertIsNone(
            _intersect_ray_with_plane(
                np.asarray((0.0, 0.0, 0.0), dtype=float),
                np.asarray((1.0, 0.0, 0.0), dtype=float),
                np.asarray((0.0, 2.0, 0.0), dtype=float),
                np.asarray((0.0, 1.0, 0.0), dtype=float),
            )
        )
        self.assertIsNone(
            _intersect_ray_with_plane(
                np.asarray((0.0, 3.0, 0.0), dtype=float),
                np.asarray((0.0, 1.0, 0.0), dtype=float),
                np.asarray((0.0, 2.0, 0.0), dtype=float),
                np.asarray((0.0, 1.0, 0.0), dtype=float),
            )
        )

    def test_axis_drag_plane_is_finite_and_perpendicular_to_axis(self) -> None:
        axis = np.asarray((1.0, 0.0, 0.0), dtype=float)

        ordinary = _build_axis_drag_plane_normal(
            axis,
            np.asarray((0.0, 1.0, 1.0), dtype=float),
        )
        degenerate = _build_axis_drag_plane_normal(
            axis,
            np.asarray((1.0, 0.0, 0.0), dtype=float),
        )

        for normal in (ordinary, degenerate):
            self.assertTrue(np.all(np.isfinite(normal)))
            self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0)
            self.assertAlmostEqual(float(np.dot(normal, axis)), 0.0)

    def test_signed_rotation_follows_the_selected_axis(self) -> None:
        axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
        positive = _get_signed_rotation_degrees(
            axis,
            np.asarray((1.0, 0.0, 0.0), dtype=float),
            np.asarray((0.0, 1.0, 0.0), dtype=float),
        )
        negative = _get_signed_rotation_degrees(
            axis,
            np.asarray((0.0, 1.0, 0.0), dtype=float),
            np.asarray((1.0, 0.0, 0.0), dtype=float),
        )

        self.assertAlmostEqual(positive, 90.0)
        self.assertAlmostEqual(negative, -90.0)

    def test_object_ray_pick_uses_nearest_target_and_fading_mirror(self) -> None:
        near = _build_placed_object("near")
        far = _build_placed_object("far", world_position=(0.0, 3.0, 0.0))
        origin, direction = _forward_ray()

        hit = _get_nearest_preview_placed_object_ray_hit(
            (far, near),
            origin,
            direction,
        )

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit[0].object_id, "near")
        np.testing.assert_allclose(hit[1], (0.0, -0.5, 0.5))
        self.assertAlmostEqual(hit[2], 4.5)

        symmetric = _build_placed_object("half", symmetric=True)
        mirror_origin, mirror_direction = _forward_ray(x=-0.4)
        mirrored_hit = _get_nearest_preview_placed_object_ray_hit(
            (symmetric,),
            mirror_origin,
            mirror_direction,
        )

        self.assertIsNotNone(mirrored_hit)
        assert mirrored_hit is not None
        self.assertEqual(mirrored_hit[0].object_id, "half")


# ### Canvas object gizmo tests ###
class CanvasObjectGizmoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.exit_first_person_mode()
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(
        self,
        *placed_objects: PreviewPlacedObject,
    ) -> GlbViewerWidget:
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_model(_build_preview_model(*placed_objects))
        self.widgets.append(viewer)
        return viewer

    def test_nearest_canvas_hit_selects_exactly_one_object_or_wall(self) -> None:
        wall = _build_wall(y=2.0)
        viewer = self._build_viewer(_build_placed_object("chair"))
        viewer.set_wall_targets((wall,))
        ray = _forward_ray()

        with patch.object(viewer.view, "build_camera_ray", return_value=ray):
            viewer.view.viewport_clicked.emit(QPointF(20.0, 20.0))

        self.assertEqual(viewer.get_selected_placed_object_id(), "chair")
        self.assertIsNone(viewer.get_selected_wall_surface_id())

        viewer.set_model(
            _build_preview_model(
                _build_placed_object(
                    "chair",
                    world_position=(0.0, 3.0, 0.0),
                )
            ),
            preserve_camera=True,
        )
        with patch.object(viewer.view, "build_camera_ray", return_value=ray):
            viewer.view.viewport_clicked.emit(QPointF(20.0, 20.0))

        self.assertIsNone(viewer.get_selected_placed_object_id())
        self.assertEqual(viewer.get_selected_wall_surface_id(), wall.surface_id)

    def test_object_selection_survives_refresh_until_target_disappears(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))

        self.assertTrue(viewer.select_placed_object("chair"))
        viewer.set_model(
            _build_preview_model(
                _build_placed_object(
                    "chair",
                    world_position=(2.0, 1.0, 0.0),
                )
            ),
            preserve_camera=True,
        )

        self.assertEqual(viewer.get_selected_placed_object_id(), "chair")
        viewer.set_model(_build_preview_model(), preserve_camera=True)
        self.assertIsNone(viewer.get_selected_placed_object_id())

    def test_wall_and_window_tools_clear_object_selection(self) -> None:
        wall = _build_wall()
        viewer = self._build_viewer(_build_placed_object("chair"))
        viewer.set_wall_targets((wall,))
        viewer.select_placed_object("chair")

        self.assertTrue(viewer.select_wall_target(wall.surface_id))
        self.assertIsNone(viewer.get_selected_placed_object_id())
        self.assertTrue(viewer.begin_window_placement())
        self.assertIsNone(viewer.get_selected_placed_object_id())

    def test_delete_key_requests_only_selected_placement_removal(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        placement_removals: list[str] = []
        generic_deletions: list[bool] = []
        viewer.placed_object_removal_requested.connect(
            placement_removals.append
        )
        viewer.delete_requested.connect(lambda: generic_deletions.append(True))
        viewer.select_placed_object("chair")

        viewer.view.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Delete,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.assertEqual(placement_removals, ["chair"])
        self.assertEqual(generic_deletions, [])
        self.assertIsNone(viewer.get_selected_placed_object_id())

        viewer.view.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Delete,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertEqual(generic_deletions, [True])

    def test_ctrl_first_person_pointer_can_select_an_object(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        position = QPointF(40.0, 24.0)
        viewer.enter_first_person_mode()
        QTest.keyPress(viewer.view, Qt.Key.Key_Control)

        with (
            patch.object(viewer.view, "build_camera_ray", return_value=_forward_ray()),
            patch.object(viewer.view, "_get_clicked_items", return_value=[]),
        ):
            viewer.view.mousePressEvent(
                _MouseButtonEvent(
                    button=Qt.MouseButton.LeftButton,
                    position=position,
                )
            )
            viewer.view.mouseReleaseEvent(
                _MouseButtonEvent(
                    button=Qt.MouseButton.LeftButton,
                    position=position,
                )
            )
        QTest.keyRelease(viewer.view, Qt.Key.Key_Control)

        self.assertEqual(viewer.get_selected_placed_object_id(), "chair")
        self.assertEqual(viewer.get_navigation_mode(), NAVIGATION_MODE_FIRST_PERSON)
        self.assertTrue(viewer.is_first_person_pointer_captured)

    def test_translation_drag_emits_one_world_transform_on_release(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        viewer.select_placed_object("chair")
        emitted: list[tuple[str, object, object]] = []
        viewer.placed_object_transform_changed.connect(
            lambda object_id, position, rotation: emitted.append(
                (object_id, position, rotation)
            )
        )
        handle = _TransformGizmoHandle(kind="translate", axis_index=0)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_forward_ray(x=0.0, z=0.0),
        ):
            self.assertTrue(
                viewer._begin_placed_object_gizmo_drag(handle, QPointF())
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_forward_ray(x=2.0, z=0.0),
        ):
            self.assertTrue(
                viewer._update_placed_object_gizmo_drag(QPointF(20.0, 0.0))
            )
            self.assertTrue(
                viewer._finish_placed_object_gizmo_drag(QPointF(20.0, 0.0))
            )

        self.assertEqual(len(emitted), 1)
        object_id, position, rotation = emitted[0]
        self.assertEqual(object_id, "chair")
        np.testing.assert_allclose(position, (2.0, 0.0, 0.0), atol=1e-7)
        np.testing.assert_allclose(rotation, (0.0, 0.0, 0.0), atol=1e-7)

        group = viewer._placed_object_render_groups["chair"]
        self.assertEqual(group.preview.world_position, (2.0, 0.0, 0.0))
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_forward_ray(x=2.0, z=0.0),
        ):
            self.assertTrue(
                viewer._begin_placed_object_gizmo_drag(handle, QPointF())
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_forward_ray(x=3.0, z=0.0),
        ):
            self.assertTrue(
                viewer._finish_placed_object_gizmo_drag(QPointF(30.0, 0.0))
            )

        self.assertEqual(len(emitted), 2)
        np.testing.assert_allclose(emitted[1][1], (3.0, 0.0, 0.0), atol=1e-7)

    def test_rotation_drag_emits_signed_axis_rotation_on_release(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        viewer.select_placed_object("chair")
        emitted: list[tuple[str, object, object]] = []
        viewer.placed_object_transform_changed.connect(
            lambda object_id, position, rotation: emitted.append(
                (object_id, position, rotation)
            )
        )
        handle = _TransformGizmoHandle(kind="rotate", axis_index=2)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_downward_ray(1.0, 0.0),
        ):
            self.assertTrue(
                viewer._begin_placed_object_gizmo_drag(handle, QPointF())
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_downward_ray(0.0, 1.0),
        ):
            self.assertTrue(
                viewer._update_placed_object_gizmo_drag(QPointF(20.0, 20.0))
            )
            self.assertTrue(
                viewer._finish_placed_object_gizmo_drag(QPointF(20.0, 20.0))
            )

        self.assertEqual(len(emitted), 1)
        object_id, position, rotation = emitted[0]
        self.assertEqual(object_id, "chair")
        np.testing.assert_allclose(position, (0.0, 0.0, 0.0), atol=1e-7)
        np.testing.assert_allclose(rotation, (0.0, 0.0, 90.0), atol=1e-7)

    def test_cancelled_drag_restores_preview_and_emits_nothing(self) -> None:
        viewer = self._build_viewer(_build_placed_object("chair"))
        viewer.select_placed_object("chair")
        group = viewer._placed_object_render_groups["chair"]
        original_transform = group.current_transform.copy()
        emitted: list[tuple[str, object, object]] = []
        viewer.placed_object_transform_changed.connect(
            lambda object_id, position, rotation: emitted.append(
                (object_id, position, rotation)
            )
        )
        handle = _TransformGizmoHandle(kind="translate", axis_index=0)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_forward_ray(x=0.0, z=0.0),
        ):
            self.assertTrue(
                viewer._begin_placed_object_gizmo_drag(handle, QPointF())
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_forward_ray(x=2.0, z=0.0),
        ):
            self.assertTrue(
                viewer._update_placed_object_gizmo_drag(QPointF(20.0, 0.0))
            )
        self.assertFalse(np.allclose(group.current_transform, original_transform))
        self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)

        viewer.view.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.assertEqual(emitted, [])
        self.assertEqual(viewer.get_selected_placed_object_id(), "chair")
        np.testing.assert_allclose(group.current_transform, original_transform)
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_half_object_and_fading_mirror_share_the_live_transform_root(
        self,
    ) -> None:
        viewer = self._build_viewer(_build_placed_object("half", symmetric=True))
        group = viewer._placed_object_render_groups["half"]

        self.assertEqual(len(group.retained_parts), 1)
        self.assertEqual(len(group.symmetric_groups), 1)
        retained = group.retained_parts[0]
        mirrored = group.symmetric_groups[0]
        self.assertIs(retained.mesh_item.parentItem(), group.root_item)
        self.assertIs(mirrored.mesh_item.parentItem(), group.root_item)
        if retained.textured_item is not None:
            self.assertIs(retained.textured_item.parentItem(), group.root_item)
        if mirrored.textured_item is not None:
            self.assertIs(mirrored.textured_item.parentItem(), group.root_item)

    def test_placed_prefab_glass_and_its_mirror_keep_sidedness(self) -> None:
        glass_mesh = trimesh.Trimesh(
            vertices=np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                dtype=float,
            ),
            faces=np.asarray(((0, 1, 2),), dtype=np.int64),
            process=False,
        )
        glass_mesh.visual = TextureVisuals(
            uv=np.zeros((3, 2), dtype=float),
            material=build_housemaker_glass_material(False),
        )
        preview = PreviewPlacedObject(
            object_id="glass-half",
            meshes=(glass_mesh,),
            placement_transform=np.eye(4, dtype=float),
            world_position=(0.0, 0.0, 0.0),
            rotation_degrees=(0.0, 0.0, 0.0),
            symmetric_preview_orientation="vertical",
            symmetric_preview_plane_coordinate=0.0,
        )

        viewer = self._build_viewer(preview)
        group = viewer._placed_object_render_groups["glass-half"]

        retained = group.retained_parts[0].textured_item
        mirrored = group.symmetric_groups[0].textured_item
        self.assertIsNotNone(retained)
        self.assertIsNotNone(mirrored)
        assert retained is not None
        assert mirrored is not None
        self.assertTrue(retained._is_prefab_glass)
        self.assertTrue(mirrored._is_prefab_glass)
        self.assertFalse(retained._double_sided)
        self.assertFalse(mirrored._double_sided)


if __name__ == "__main__":
    unittest.main()
