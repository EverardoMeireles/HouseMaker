# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import GeneratedModel, PreviewPlacedObject
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData
from housemaker.surface_geometry import SURFACE_TYPE_WALL, FixedSurface

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _placed_object(
    object_id: str,
    *,
    x: float,
) -> PreviewPlacedObject:
    mesh = trimesh.creation.box(extents=(0.75, 0.75, 0.75))
    mesh.apply_translation((0.0, 0.0, 0.375))
    transform = np.eye(4, dtype=float)
    transform[0, 3] = x
    return PreviewPlacedObject(
        object_id=object_id,
        meshes=(mesh,),
        placement_transform=transform,
        world_position=(x, 0.0, 0.0),
        rotation_degrees=(0.0, 0.0, 0.0),
    )


def _preview_model(*object_ids: str) -> GeneratedModel:
    base_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    base_mesh.apply_translation((100.0, 100.0, 100.0))
    placed_objects = [
        _placed_object(object_id, x=float(index * 2))
        for index, object_id in enumerate(object_ids)
    ]
    return GeneratedModel(
        mesh=base_mesh,
        scene=trimesh.Scene(base_mesh.copy()),
        glb_bytes=b"",
        preview_placed_objects=placed_objects,
        preview_base_mesh=base_mesh.copy(),
    )


def _wall() -> FixedSurface:
    vertices = np.asarray(
        (
            (-1.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
            (1.0, 2.0, 2.0),
            (-1.0, 2.0, 2.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id="level:0/wall:1:2",
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=4.0,
        wall_key="1:2",
        wall_start_world=(-1.0, 2.0, 0.0),
        wall_end_world=(1.0, 2.0, 0.0),
        wall_height_meters=2.0,
    )


def _empty_level() -> LevelData:
    return LevelData(
        index=0,
        name="Ground",
        vertex_data=VertexData(),
    )


# ### Main-workspace multi-object selection tests ###
class CanvasMultiObjectSelectionMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1200, 760)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _select_two_objects(self, *, active_object_id: str) -> None:
        viewer = self.workspace.viewer
        viewer.set_model(_preview_model("chair", "table"))
        self.assertTrue(
            viewer.set_selected_placed_object_ids(
                ("chair", "table"),
                active_object_id=active_object_id,
            )
        )
        self.assertEqual(
            self.workspace._desired_canvas_object_ids,
            ("chair", "table"),
        )
        self.assertEqual(
            self.workspace._desired_canvas_object_id,
            active_object_id,
        )

    def test_full_selection_and_active_object_restore_after_preview_rebuild(
        self,
    ) -> None:
        self._select_two_objects(active_object_id="chair")
        viewer = self.workspace.viewer

        self.workspace._is_syncing_canvas_scene_selection = True
        try:
            viewer.clear_model()
            viewer.set_model(
                _preview_model("chair", "table"),
                preserve_camera=True,
            )
        finally:
            self.workspace._is_syncing_canvas_scene_selection = False
        self.assertEqual(viewer.get_selected_placed_object_ids(), ())

        self.workspace._restore_desired_canvas_scene_selection()

        self.assertTrue(viewer._last_set_model_preserved_camera)
        self.assertEqual(
            viewer.get_selected_placed_object_ids(),
            ("chair", "table"),
        )
        self.assertEqual(viewer.get_selected_placed_object_id(), "chair")
        self.assertEqual(
            self.workspace._desired_canvas_object_ids,
            ("chair", "table"),
        )
        self.assertEqual(self.workspace._desired_canvas_object_id, "chair")

    def test_surface_selection_clears_the_remembered_object_set(self) -> None:
        self._select_two_objects(active_object_id="table")
        wall = _wall()
        self.workspace._set_canvas_viewer_targets((wall,))

        self.assertTrue(
            self.workspace.viewer.set_selected_canvas_surface_ids(
                (wall.surface_id,)
            )
        )

        self.assertEqual(self.workspace._desired_canvas_object_ids, ())
        self.assertIsNone(self.workspace._desired_canvas_object_id)
        self.assertEqual(
            self.workspace._desired_canvas_surface_ids,
            (wall.surface_id,),
        )

    def test_discarding_one_selected_object_retains_the_other_selection(
        self,
    ) -> None:
        self._select_two_objects(active_object_id="chair")

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace._handle_generated_object_deleted_for_canvas("chair")

        self.assertEqual(
            self.workspace._desired_canvas_object_ids,
            ("table",),
        )
        self.assertEqual(self.workspace._desired_canvas_object_id, "table")
        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_project_state_reset_clears_multi_object_selection(self) -> None:
        self._select_two_objects(active_object_id="table")

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ):
            self.workspace._apply_project_state([_empty_level()], 0)

        self.assertEqual(self.workspace._desired_canvas_object_ids, ())
        self.assertIsNone(self.workspace._desired_canvas_object_id)
        self.assertEqual(self.workspace._desired_canvas_surface_ids, ())


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
