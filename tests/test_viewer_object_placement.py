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
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.surface_geometry import (
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    FixedSurface,
)
from housemaker.viewer import (
    GlbViewerWidget,
    SceneObjectPlacementCandidate,
    build_scene_object_placement_group_candidate,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _horizontal_floor(level_index: int = 2) -> FixedSurface:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((-2.0, -2.0, 0.2), (2.0, -2.0, 0.2), (2.0, 2.0, 0.2), (-2.0, 2.0, 0.2)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=f"level:{level_index}/floor",
        surface_type=SURFACE_TYPE_FLOOR,
        level_index=level_index,
        room_index=None,
        mesh=mesh,
        area_square_meters=16.0,
    )


def _blocking_wall(level_index: int = 2) -> FixedSurface:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((-2.0, -1.0, 0.0), (2.0, -1.0, 0.0), (2.0, -1.0, 3.0), (-2.0, -1.0, 3.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=f"level:{level_index}/wall",
        surface_type=SURFACE_TYPE_WALL,
        level_index=level_index,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
    )


# ### Direct-placement tests ###
class ViewerObjectPlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = GlbViewerWidget(window_editing_enabled=True)
        self.viewer.set_wall_targets((_horizontal_floor(),))

    def tearDown(self) -> None:
        self.viewer.close()
        self.viewer.deleteLater()
        _qt_application.processEvents()

    def test_cube_group_preview_and_commit_share_ordered_positions(self) -> None:
        self.assertTrue(
            self.viewer.begin_object_placement("batch", preview_count=3)
        )
        selected = QSignalSpy(self.viewer.object_placement_selected)
        ray = (
            np.asarray((0.0, 0.0, 5.0), dtype=float),
            np.asarray((0.0, 0.0, -1.0), dtype=float),
        )
        with patch.object(self.viewer.view, "build_camera_ray", return_value=ray):
            self.viewer._update_object_placement_hover(QPointF(50.0, 50.0))

        candidate = self.viewer._object_placement_candidate
        self.assertIsInstance(candidate, SceneObjectPlacementCandidate)
        assert candidate is not None
        self.assertEqual(candidate.level_index, 2)
        self.assertEqual(len(candidate.world_positions), 3)
        self.assertEqual(len(set(candidate.world_positions)), 3)
        self.assertTrue(self.viewer._commit_object_placement())
        self.assertEqual(selected.count(), 1)
        self.assertEqual(selected.at(0)[0], "batch")
        self.assertEqual(selected.at(0)[1], candidate)
        self.assertFalse(self.viewer.is_object_placement_active)

    def test_non_floor_nearest_hit_is_not_a_valid_candidate(self) -> None:
        self.viewer.set_wall_targets((_horizontal_floor(), _blocking_wall()))
        self.assertTrue(self.viewer.begin_object_placement("blocked"))
        ray = (
            np.asarray((0.0, -2.0, 1.0), dtype=float),
            np.asarray((0.0, 1.0, -0.2), dtype=float),
        )
        with patch.object(self.viewer.view, "build_camera_ray", return_value=ray):
            self.viewer._update_object_placement_hover(QPointF())

        self.assertIsNone(self.viewer._object_placement_candidate)
        self.assertFalse(self.viewer._commit_object_placement())

    def test_actual_mesh_preview_is_copied_without_replacing_its_shape(self) -> None:
        source = trimesh.creation.icosphere(radius=0.6)
        source_vertices = np.asarray(source.vertices).copy()

        self.assertTrue(
            self.viewer.begin_object_placement(
                "atlas-object",
                preview_meshes=(source,),
            )
        )

        self.assertEqual(len(self.viewer._object_placement_preview_meshes), 1)
        np.testing.assert_allclose(source.vertices, source_vertices)
        np.testing.assert_allclose(
            self.viewer._object_placement_preview_meshes[0].extents,
            source.extents,
        )
        preview_bounds = self.viewer._object_placement_preview_meshes[0].bounds
        self.assertAlmostEqual(float(preview_bounds[0, 2]), 0.0)
        np.testing.assert_allclose(
            np.mean(preview_bounds[:, :2], axis=0),
            (0.0, 0.0),
        )

    def test_public_group_builder_centers_members_without_overlap(self) -> None:
        candidate = build_scene_object_placement_group_candidate(
            4,
            (10.0, 20.0, 3.0),
            4,
        )

        self.assertEqual(candidate.level_index, 4)
        self.assertEqual(len(candidate.world_positions), 4)
        self.assertTrue(all(position[2] == 3.0 for position in candidate.world_positions))
        mean = np.mean(np.asarray(candidate.world_positions), axis=0)
        np.testing.assert_allclose(mean, (10.0, 20.0, 3.0))


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
