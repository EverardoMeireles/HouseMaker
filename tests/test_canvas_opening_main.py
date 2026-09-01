# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication
import trimesh

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_openings import (
    CANVAS_OPENING_DOORWAY,
    CanvasOpeningBounds,
    CanvasOpeningEdit,
    build_canvas_opening_targets,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import DoorwayData, LevelData, WindowData
from housemaker.surface_geometry import FixedSurface, SURFACE_TYPE_WALL


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level_and_wall() -> tuple[LevelData, FixedSurface]:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (4.0, 0.0, 3.0),
            (0.0, 0.0, 3.0),
        ),
        dtype=float,
    )
    wall = FixedSurface(
        surface_id="level:2/wall:1:2",
        surface_type=SURFACE_TYPE_WALL,
        level_index=2,
        room_index=None,
        mesh=trimesh.Trimesh(
            vertices=vertices,
            faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
            process=False,
        ),
        area_square_meters=12.0,
        wall_key="1:2",
        wall_start_world=(0.0, 0.0, 0.0),
        wall_end_world=(4.0, 0.0, 0.0),
        wall_height_meters=3.0,
    )
    level = LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        doorways=[
            DoorwayData(
                center_x=100.0,
                center_y=0.0,
                width_meters=1.0,
                height_meters=2.0,
                rotation_degrees=90.0,
            )
        ],
        windows=[
            WindowData(
                window_id="editable-window",
                wall_surface_id=wall.surface_id,
                start_ratio=0.6,
                end_ratio=0.8,
                bottom_ratio=0.3,
                top_ratio=0.7,
            )
        ],
    )
    return level, wall


# ### Main opening edit integration tests ###
class CanvasOpeningMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        level, wall = _build_level_and_wall()
        self.level = level
        self.wall = wall
        self.workspace.levels = [level]
        self.workspace.current_level_index = 0
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_canvas_to_current_level()
        self.workspace._set_canvas_viewer_targets((wall,))

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _get_target(self, kind: str):
        return next(
            target
            for target in build_canvas_opening_targets(
                (self.level,),
                (self.wall,),
            )
            if target.reference.kind == kind
        )

    def test_doorway_drag_keeps_the_old_mesh_until_release_delay(self) -> None:
        target = self._get_target(CANVAS_OPENING_DOORWAY)
        edit = CanvasOpeningEdit(
            target.reference,
            target.wall_surface_id,
            CanvasOpeningBounds(0.2, 0.65, 0.15, 0.85),
        )
        committed_before = self.workspace._viewer_doorways_by_level_index[2]

        self.workspace._handle_canvas_opening_edit_started(
            CanvasOpeningEdit(
                target.reference,
                target.wall_surface_id,
                target.bounds,
            )
        )
        self.workspace._handle_canvas_opening_edit_preview_changed(edit)

        self.assertTrue(self.workspace._is_canvas_opening_drag_active)
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertEqual(
            self.workspace._viewer_doorways_by_level_index[2],
            committed_before,
        )
        self.assertNotEqual(tuple(self.level.doorways), committed_before)

        self.workspace._handle_canvas_opening_edit_finished(edit, True)

        self.assertFalse(self.workspace._is_canvas_opening_drag_active)
        self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace._commit_pending_doorway_mesh_update()

        self.assertEqual(
            self.workspace._viewer_doorways_by_level_index[2],
            tuple(self.level.doorways),
        )
        schedule_refresh.assert_called_once_with(preserve_camera=True)

    def test_cancelled_window_drag_restores_data_without_a_mesh_update(self) -> None:
        target = next(
            target
            for target in build_canvas_opening_targets(
                (self.level,),
                (self.wall,),
            )
            if target.reference.stable_id == "editable-window"
        )
        starting_edit = CanvasOpeningEdit(
            target.reference,
            target.wall_surface_id,
            target.bounds,
        )
        changed_edit = CanvasOpeningEdit(
            target.reference,
            target.wall_surface_id,
            CanvasOpeningBounds(0.4, 0.75, 0.15, 0.85),
        )
        original_window = self.level.windows[0]

        self.workspace._handle_canvas_opening_edit_started(starting_edit)
        self.workspace._handle_canvas_opening_edit_preview_changed(changed_edit)
        self.assertNotEqual(self.level.windows[0], original_window)

        self.workspace._handle_canvas_opening_edit_cancelled(starting_edit)

        self.assertEqual(self.level.windows[0], original_window)
        self.assertIsNone(self.workspace._pending_window_mesh_level_index)
        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertFalse(self.workspace._is_canvas_opening_drag_active)

    def test_switching_openings_stages_one_refresh_after_the_second_drag(
        self,
    ) -> None:
        doorway_target = self._get_target(CANVAS_OPENING_DOORWAY)
        window_target = next(
            target
            for target in build_canvas_opening_targets(
                (self.level,),
                (self.wall,),
            )
            if target.reference.stable_id == "editable-window"
        )
        doorway_start = CanvasOpeningEdit(
            doorway_target.reference,
            doorway_target.wall_surface_id,
            doorway_target.bounds,
        )
        doorway_edit = CanvasOpeningEdit(
            doorway_target.reference,
            doorway_target.wall_surface_id,
            CanvasOpeningBounds(0.25, 0.65, 0.0, 0.75),
        )
        window_start = CanvasOpeningEdit(
            window_target.reference,
            window_target.wall_surface_id,
            window_target.bounds,
        )
        window_edit = CanvasOpeningEdit(
            window_target.reference,
            window_target.wall_surface_id,
            CanvasOpeningBounds(0.55, 0.85, 0.25, 0.75),
        )

        self.workspace._handle_canvas_opening_edit_started(doorway_start)
        self.workspace._handle_canvas_opening_edit_preview_changed(doorway_edit)
        self.workspace._handle_canvas_opening_edit_finished(doorway_edit, True)

        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace._handle_canvas_opening_edit_started(window_start)

            schedule_refresh.assert_not_called()
            self.assertTrue(
                self.workspace._staged_canvas_opening_mesh_update
            )
            self.assertTrue(self.workspace._is_canvas_opening_drag_active)
            self.assertFalse(
                self.workspace._doorway_mesh_update_timer.isActive()
            )

            self.workspace._handle_canvas_opening_edit_preview_changed(
                window_edit
            )
            self.workspace._handle_canvas_opening_edit_finished(
                window_edit,
                True,
            )
            self.workspace._commit_pending_doorway_mesh_update()

        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertEqual(
            self.workspace._viewer_doorways_by_level_index[2],
            tuple(self.level.doorways),
        )
        self.assertEqual(
            self.workspace._viewer_windows_by_level_index[2],
            tuple(self.level.windows),
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
