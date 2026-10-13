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

from housemaker.glb import GeneratedModel, PreviewPlacedObject
from housemaker.viewer import GlassAreaPlacementRequest, GlbViewerWidget

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_body_preview() -> PreviewPlacedObject:
    mesh = trimesh.creation.box(extents=(2.0, 0.2, 4.0))
    mesh.apply_translation((0.0, 0.0, 2.0))
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (10.0, 5.0, 3.0)
    return PreviewPlacedObject(
        object_id="door-a:body",
        meshes=(mesh,),
        placement_transform=transform,
        world_position=(10.0, 5.0, 3.0),
        rotation_degrees=(0.0, 0.0, 0.0),
    )


def _build_half_body_preview() -> PreviewPlacedObject:
    """Build the retained left half of the fixture's full two-meter profile."""

    mesh = trimesh.creation.box(extents=(1.0, 0.2, 4.0))
    mesh.apply_translation((-0.5, 0.0, 2.0))
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (10.0, 5.0, 3.0)
    return PreviewPlacedObject(
        object_id="door-a:body",
        meshes=(mesh,),
        placement_transform=transform,
        world_position=(10.0, 5.0, 3.0),
        rotation_degrees=(0.0, 0.0, 0.0),
    )


def _build_model(preview: PreviewPlacedObject) -> GeneratedModel:
    base_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    base_mesh.apply_translation((100.0, 100.0, 100.0))
    return GeneratedModel(
        mesh=base_mesh,
        scene=trimesh.Scene(base_mesh),
        glb_bytes=b"",
        preview_placed_objects=[preview],
        preview_base_mesh=base_mesh.copy(),
    )


def _front_ray(world_x: float, world_z: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((world_x, 0.0, world_z), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


# ### Glass-area placement tests ###
class ViewerGlassAreaPlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = GlbViewerWidget(placed_object_editing_enabled=True)
        self.viewer.set_model(_build_model(_build_body_preview()))

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def test_drag_emits_normalized_local_xz_bounds_and_live_outline(self) -> None:
        emitted: list[GlassAreaPlacementRequest] = []
        self.viewer.glass_area_placement_requested.connect(emitted.append)
        self.assertTrue(
            self.viewer.begin_glass_area_placement(
                "door",
                "door-a",
                "door-a:body",
            )
        )
        self.assertTrue(self.viewer.is_glass_area_placement_active)
        self.assertFalse(self.viewer.is_window_placement_active())

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            side_effect=(
                _front_ray(9.5, 4.0),
                _front_ray(10.5, 6.0),
                _front_ray(10.5, 6.0),
            ),
        ):
            self.viewer._handle_window_pointer_pressed(QPointF(10.0, 10.0))
            self.viewer._handle_window_pointer_moved(QPointF(40.0, 40.0))
            self.assertIsNotNone(self.viewer._glass_area_preview_item)
            self.viewer._handle_window_pointer_released(QPointF(40.0, 40.0))

        self.assertEqual(len(emitted), 1)
        request = emitted[0]
        self.assertEqual(request.asset_kind, "door")
        self.assertEqual(request.asset_id, "door-a")
        self.assertEqual(request.body_object_id, "door-a:body")
        np.testing.assert_allclose(request.normalized_x_bounds, (0.25, 0.75))
        np.testing.assert_allclose(request.normalized_z_bounds, (0.25, 0.75))
        self.assertFalse(self.viewer.is_glass_area_placement_active)
        self.assertIsNone(self.viewer._glass_area_preview_item)

    def test_cancel_is_separate_from_window_frame_rectangle_placement(self) -> None:
        cancelled: list[tuple[str, str]] = []
        self.viewer.glass_area_placement_cancelled.connect(
            lambda kind, asset_id: cancelled.append((kind, asset_id))
        )
        self.assertFalse(
            self.viewer.begin_glass_area_placement(
                "window",
                "window-a",
                "missing-body",
            )
        )
        self.assertTrue(
            self.viewer.begin_glass_area_placement(
                "window",
                "window-a",
                "door-a:body",
            )
        )

        self.viewer._handle_rectangle_drawing_cancel_requested()

        self.assertEqual(cancelled, [("window", "window-a")])
        self.assertFalse(self.viewer.is_glass_area_placement_active)

    def test_half_mesh_drag_uses_the_reconstructed_full_profile(self) -> None:
        self.viewer.set_model(_build_model(_build_half_body_preview()))
        emitted: list[GlassAreaPlacementRequest] = []
        self.viewer.glass_area_placement_requested.connect(emitted.append)
        self.assertTrue(
            self.viewer.begin_glass_area_placement(
                "door",
                "door-a",
                "door-a:body",
                profile_bounds=(-1.0, 1.0, 0.0, 4.0),
            )
        )

        with patch.object(
            self.viewer.view,
            "build_camera_ray",
            side_effect=(
                _front_ray(9.25, 4.0),
                _front_ray(9.75, 6.0),
            ),
        ):
            self.viewer._handle_window_pointer_pressed(QPointF(10.0, 10.0))
            self.viewer._handle_window_pointer_released(QPointF(40.0, 40.0))

        self.assertEqual(len(emitted), 1)
        np.testing.assert_allclose(
            emitted[0].normalized_x_bounds,
            (0.125, 0.375),
        )
        np.testing.assert_allclose(
            emitted[0].normalized_z_bounds,
            (0.25, 0.75),
        )


if __name__ == "__main__":
    unittest.main()
