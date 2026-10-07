# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from housemaker.glb import (
    GeneratedModel,
    PreviewPlacedObject,
    PreviewSymmetricObject,
)
from housemaker.viewer import (
    TRANSFORM_GIZMO_SCALE,
    TRANSFORM_GIZMO_TRANSLATE,
    GlbViewerWidget,
    _TransformGizmoHandle,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _downward_ray(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((x, y, 5.0), dtype=float),
        np.asarray((0.0, 0.0, -1.0), dtype=float),
    )


def _build_side_duplicated_knob_model() -> GeneratedModel:
    knob_id = "door-1:door_knob"
    local_knob = trimesh.creation.box(extents=(0.20, 0.08, 0.20))
    initial_transform = np.eye(4, dtype=float)
    initial_transform[:3, 3] = (0.35, -0.12, 1.0)
    world_knob = local_knob.copy()
    world_knob.apply_transform(initial_transform)
    mirrored_knob = world_knob.copy()
    mirrored_vertices = np.asarray(mirrored_knob.vertices, dtype=float).copy()
    mirrored_vertices[:, 1] *= -1.0
    mirrored_knob.vertices = mirrored_vertices
    mirrored_knob.faces = np.asarray(
        mirrored_knob.faces,
        dtype=np.int64,
    )[:, (0, 2, 1)]

    base_mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    base_mesh.apply_translation((100.0, 100.0, 100.0))
    return GeneratedModel(
        mesh=base_mesh,
        scene=trimesh.Scene(base_mesh),
        glb_bytes=b"",
        preview_base_mesh=base_mesh.copy(),
        preview_placed_objects=[
            PreviewPlacedObject(
                object_id=knob_id,
                meshes=(local_knob,),
                placement_transform=initial_transform,
                world_position=(0.35, -0.12, 1.0),
                rotation_degrees=(0.0, 0.0, 0.0),
                scale=1.0,
                axis_scales=(1.0, 1.0, 1.0),
            )
        ],
        preview_symmetric_objects=[
            PreviewSymmetricObject(
                object_id=f"{knob_id}:side",
                meshes=(world_knob,),
                orientation="depth",
                plane_coordinate=0.0,
                mirrored_meshes=(mirrored_knob,),
                fade_enabled=False,
            )
        ],
    )


def _transformed_vertices(
    vertices: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    homogeneous = np.column_stack(
        (
            np.asarray(vertices, dtype=float),
            np.ones(len(vertices), dtype=float),
        )
    )
    return (homogeneous @ np.asarray(transform, dtype=float).T)[:, :3]


def _depth_reflection_transform() -> np.ndarray:
    reflection = np.eye(4, dtype=float)
    reflection[1, 1] = -1.0
    return reflection


def _item_transform(item: object) -> np.ndarray:
    return np.asarray(item.transform().matrix(), dtype=float)


# ### Side-duplicated handle live-preview tests ###
class DoorLiveMirrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = GlbViewerWidget(
            placed_object_editing_enabled=True,
            placed_object_click_selection_enabled=True,
            placed_object_auxiliary_controls_enabled=False,
            placed_object_axis_scale_gizmos_enabled=True,
            symmetric_preview_fade_enabled=False,
        )
        self.viewer.resize(640, 480)
        self.viewer.set_model(_build_side_duplicated_knob_model())
        self.viewer.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def _assert_mirror_uses_dedicated_live_root(self):
        knob_group = self.viewer._placed_object_render_groups["door-1:door_knob"]
        self.assertEqual(len(self.viewer._embedded_symmetric_preview_groups), 1)
        mirror_group = self.viewer._embedded_symmetric_preview_groups[0]
        mirror_root = mirror_group.mesh_item.parentItem()
        self.assertIsNotNone(mirror_root)
        self.assertIsNot(mirror_root, knob_group.root_item)
        if mirror_group.textured_item is not None:
            self.assertIs(
                mirror_group.textured_item.parentItem(),
                mirror_root,
            )
        np.testing.assert_allclose(_item_transform(mirror_root), np.eye(4))
        return knob_group, mirror_group, mirror_root

    def _assert_live_mirror_matches_source(
        self,
        knob_group: object,
        mirror_group: object,
        mirror_root: object,
        initial_transform: np.ndarray,
    ) -> None:
        reflection = _depth_reflection_transform()
        expected_delta = (
            reflection
            @ knob_group.current_transform
            @ np.linalg.inv(initial_transform)
            @ reflection
        )
        np.testing.assert_allclose(
            _item_transform(mirror_root),
            expected_delta,
            atol=1e-7,
        )
        retained_vertices = np.asarray(
            knob_group.pick_meshes[0].vertices,
            dtype=float,
        )
        expected_mirrored_vertices = _transformed_vertices(
            retained_vertices,
            knob_group.current_transform,
        )
        expected_mirrored_vertices[:, 1] *= -1.0
        actual_mirrored_vertices = _transformed_vertices(
            mirror_group.vertices,
            _item_transform(mirror_root),
        )
        np.testing.assert_allclose(
            actual_mirrored_vertices,
            expected_mirrored_vertices,
            atol=1e-7,
        )

    def test_depth_drag_moves_side_duplicate_oppositely_and_restores_on_cancel(
        self,
    ) -> None:
        knob_group, mirror_group, mirror_root = (
            self._assert_mirror_uses_dedicated_live_root()
        )
        self.assertTrue(self.viewer.select_placed_object("door-1:door_knob"))
        emitted: list[tuple[str, object, object]] = []
        self.viewer.placed_object_transform_changed.connect(
            lambda object_id, position, rotation: emitted.append(
                (object_id, position, rotation)
            )
        )
        initial_transform = knob_group.current_transform.copy()
        handle = _TransformGizmoHandle(TRANSFORM_GIZMO_TRANSLATE, 1)

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=_downward_ray(0.35, -0.12),
        ):
            self.assertTrue(
                self.viewer._begin_placed_object_gizmo_drag(handle, QPointF())
            )
        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=_downward_ray(0.35, 0.38),
        ):
            self.assertTrue(
                self.viewer._update_placed_object_gizmo_drag(QPointF(20.0, 0.0))
            )
            self._assert_live_mirror_matches_source(
                knob_group,
                mirror_group,
                mirror_root,
                initial_transform,
            )
            self.assertAlmostEqual(knob_group.current_transform[1, 3], 0.38)
            self.assertAlmostEqual(_item_transform(mirror_root)[1, 3], -0.5)
            self.assertEqual(emitted, [])

        self.viewer._cancel_placed_object_gizmo_drag()

        np.testing.assert_allclose(
            knob_group.current_transform,
            initial_transform,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _item_transform(mirror_root),
            np.eye(4),
            atol=1e-7,
        )
        self.assertEqual(emitted, [])

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=_downward_ray(0.35, -0.12),
        ):
            self.assertTrue(
                self.viewer._begin_placed_object_gizmo_drag(handle, QPointF())
            )
        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=_downward_ray(0.35, 0.38),
        ):
            self.assertTrue(
                self.viewer._update_placed_object_gizmo_drag(QPointF(20.0, 0.0))
            )
            self.assertEqual(emitted, [])
            self.assertTrue(
                self.viewer._finish_placed_object_gizmo_drag(QPointF(20.0, 0.0))
            )

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][0], "door-1:door_knob")
        np.testing.assert_allclose(
            emitted[0][1],
            (0.35, 0.38, 1.0),
            atol=1e-7,
        )

    def test_axis_scale_drag_scales_side_duplicate_live_and_commits_on_release(
        self,
    ) -> None:
        knob_group, mirror_group, mirror_root = (
            self._assert_mirror_uses_dedicated_live_root()
        )
        with patch.object(self.viewer.view, "pixelSize", return_value=0.01):
            self.assertTrue(self.viewer.select_placed_object("door-1:door_knob"))
            self.viewer._toggle_placed_object_gizmo_mode()
        emitted: list[tuple[str, object]] = []
        self.viewer.placed_object_axis_scales_changed.connect(
            lambda object_id, scales: emitted.append((object_id, scales))
        )
        initial_transform = knob_group.current_transform.copy()
        handle = _TransformGizmoHandle(TRANSFORM_GIZMO_SCALE, 0)

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=_downward_ray(0.0, 0.0),
        ):
            self.assertTrue(
                self.viewer._begin_placed_object_gizmo_drag(handle, QPointF())
            )
        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            return_value=_downward_ray(self.viewer._transform_gizmo_size, 0.0),
        ):
            self.assertTrue(
                self.viewer._update_placed_object_gizmo_drag(QPointF(20.0, 0.0))
            )
            self._assert_live_mirror_matches_source(
                knob_group,
                mirror_group,
                mirror_root,
                initial_transform,
            )
            self.assertEqual(emitted, [])

            self.assertTrue(
                self.viewer._finish_placed_object_gizmo_drag(QPointF(20.0, 0.0))
            )

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][0], "door-1:door_knob")
        np.testing.assert_allclose(
            emitted[0][1],
            knob_group.preview.axis_scales,
            atol=1e-7,
        )
        self.assertGreater(float(emitted[0][1][0]), 1.0)
        np.testing.assert_allclose(emitted[0][1][1:], (1.0, 1.0), atol=1e-7)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
