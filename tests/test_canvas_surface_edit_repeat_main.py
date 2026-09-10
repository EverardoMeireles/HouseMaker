# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_AXIS_X,
    CANVAS_SURFACE_EDIT_AXIS_Y,
    CANVAS_SURFACE_EDIT_WALL_VERTEX,
    CanvasSurfaceEdit,
    CanvasSurfaceEditHandleTarget,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData
from housemaker.surface_geometry import (
    SURFACE_TYPE_WALL,
    FixedSurface,
    build_fixed_surfaces,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_connected_level() -> LevelData:
    """Return two walls sharing the endpoint used by repeated edits."""

    vertex_data = VertexData()
    first = vertex_data.add_vertex(0.0, 0.0)
    shared = vertex_data.add_vertex(100.0, 0.0)
    third = vertex_data.add_vertex(100.0, 100.0)
    vertex_data.add_edge(first.id, shared.id)
    vertex_data.add_edge(shared.id, third.id)
    return LevelData(
        index=2,
        name="Ground",
        height_meters=3.0,
        vertex_data=vertex_data,
    )


def _build_edit(
    target: CanvasSurfaceEditHandleTarget,
    delta_meters: float,
) -> CanvasSurfaceEdit:
    return CanvasSurfaceEdit(
        reference=target.reference,
        surface_id=target.surface_id,
        delta_meters=delta_meters,
    )


# ### Repeated wall-edit integration tests ###
class RepeatedCanvasSurfaceEditMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        settings_path = (
            Path(self._temporary_directory.name) / "settings.json"
        )
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(settings_path)
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.canvas_viewer_workspace
        )
        self.workspace.canvas_viewer_tabs.setCurrentIndex(
            self.workspace.canvas_3d_view_tab_index
        )
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _install_level(self) -> tuple[LevelData, FixedSurface]:
        level = _build_connected_level()
        self.workspace.viewer.cancel_canvas_surface_edit()
        self.workspace.levels = [level]
        self.workspace.current_level_index = 0
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_level_controls()
        self.workspace._sync_canvas_to_current_level()
        surfaces = tuple(build_fixed_surfaces([level]))
        self.workspace._set_canvas_viewer_targets(surfaces)
        self.workspace._canvas_viewer_preview_revision = (
            self.workspace._viewer_preview_revision
        )
        wall = next(
            surface
            for surface in surfaces
            if surface.surface_type == SURFACE_TYPE_WALL
            and surface.wall_key == "1:2"
        )
        self.assertTrue(
            self.workspace.viewer.select_canvas_surface_target(
                wall.surface_id
            )
        )
        return level, wall

    def _target(
        self,
        surface_id: str,
        *,
        axis_index: int,
        vertex_id: int = 2,
    ) -> CanvasSurfaceEditHandleTarget:
        return next(
            target
            for target in self.workspace._canvas_surface_edit_targets_by_key.values()
            if target.surface_id == surface_id
            and target.reference.kind == CANVAS_SURFACE_EDIT_WALL_VERTEX
            and target.reference.vertex_id == vertex_id
            and target.reference.axis_index == axis_index
        )

    def _finish_changed_drag(
        self,
        target: CanvasSurfaceEditHandleTarget,
        delta_meters: float,
    ) -> None:
        edit = _build_edit(target, delta_meters)
        self.workspace._handle_canvas_surface_edit_started(
            _build_edit(target, 0.0)
        )
        self.workspace._handle_canvas_surface_edit_preview_changed(edit)
        self.workspace._handle_canvas_surface_edit_finished(edit, True)

    def _assert_four_current_handles(
        self,
        level: LevelData,
        wall: FixedSurface,
    ) -> None:
        wall_targets = tuple(
            target
            for target in self.workspace._canvas_surface_edit_targets_by_key.values()
            if target.surface_id == wall.surface_id
        )
        self.assertEqual(len(wall_targets), 4)
        self.assertEqual(
            len(
                self.workspace.viewer._get_active_canvas_surface_edit_targets()
            ),
            4,
        )
        shared = level.vertex_data.get_vertex(2)
        assert shared is not None
        expected_world = level_image_to_world_xy(level, shared.x, shared.y)
        shared_targets = tuple(
            target
            for target in wall_targets
            if target.reference.vertex_id == shared.id
        )
        self.assertEqual(len(shared_targets), 2)
        for target in shared_targets:
            self.assertAlmostEqual(target.origin_world[0], expected_world[0])
            self.assertAlmostEqual(target.origin_world[1], expected_world[1])
            self.assertEqual(
                target.chain_vertex_image_positions[-1],
                (shared.x, shared.y),
            )

    def test_first_release_retains_current_handles_and_defers_validation(
        self,
    ) -> None:
        level, wall = self._install_level()
        target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )

        with (
            patch(
                "housemaker.canvas_surface_edits._validate_generated_surfaces"
            ) as validate_geometry,
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
            ) as build_preview,
        ):
            self._finish_changed_drag(target, 0.5)

        shared = level.vertex_data.get_vertex(2)
        assert shared is not None
        self.assertAlmostEqual(shared.x, 125.0)
        self.assertTrue(self.workspace._pending_canvas_surface_mesh_update)
        self.assertTrue(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        self._assert_four_current_handles(level, wall)
        validate_geometry.assert_not_called()
        reconcile_assignments.assert_not_called()
        schedule_refresh.assert_not_called()
        build_preview.assert_not_called()

    def test_second_drag_accumulates_and_restarts_one_debounce(self) -> None:
        level, wall = self._install_level()
        first_target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )

        with (
            patch(
                "housemaker.canvas_surface_edits._validate_generated_surfaces"
            ) as validate_geometry,
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            self._finish_changed_drag(first_target, 0.5)
            second_target = self._target(
                wall.surface_id,
                axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
            )
            self.assertIsNot(second_target, first_target)

            self.workspace._handle_canvas_surface_edit_started(
                _build_edit(second_target, 0.0)
            )
            self.assertFalse(
                self.workspace._canvas_surface_mesh_update_timer.isActive()
            )
            second_edit = _build_edit(second_target, 0.25)
            self.workspace._handle_canvas_surface_edit_preview_changed(
                second_edit
            )
            self.workspace._handle_canvas_surface_edit_finished(
                second_edit,
                True,
            )

            shared = level.vertex_data.get_vertex(2)
            assert shared is not None
            self.assertAlmostEqual(shared.x, 137.5)
            self.assertTrue(
                self.workspace._canvas_surface_mesh_update_timer.isActive()
            )
            self._assert_four_current_handles(level, wall)
            validate_geometry.assert_not_called()
            reconcile_assignments.assert_not_called()
            schedule_refresh.assert_not_called()

            self.workspace._commit_pending_canvas_surface_mesh_update()

        validate_geometry.assert_called_once()
        reconcile_assignments.assert_called_once_with([level])
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        shared = level.vertex_data.get_vertex(2)
        assert shared is not None
        self.assertAlmostEqual(shared.x, 137.5)
        self.assertFalse(self.workspace._pending_canvas_surface_mesh_update)
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )

    def test_no_change_and_cancelled_followups_restart_pending_timer(
        self,
    ) -> None:
        level, wall = self._install_level()
        first_target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        self._finish_changed_drag(first_target, 0.5)

        unchanged_target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        self.workspace._handle_canvas_surface_edit_started(
            _build_edit(unchanged_target, 0.0)
        )
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        self.workspace._handle_canvas_surface_edit_finished(
            _build_edit(unchanged_target, 0.0),
            False,
        )
        self.assertTrue(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )

        cancelled_target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        self.workspace._handle_canvas_surface_edit_started(
            _build_edit(cancelled_target, 0.0)
        )
        self.workspace._handle_canvas_surface_edit_preview_changed(
            _build_edit(cancelled_target, 0.2)
        )
        shared = level.vertex_data.get_vertex(2)
        assert shared is not None
        self.assertAlmostEqual(shared.x, 135.0)
        self.workspace._handle_canvas_surface_edit_cancelled(
            _build_edit(cancelled_target, 0.0)
        )

        shared = level.vertex_data.get_vertex(2)
        assert shared is not None
        self.assertAlmostEqual(shared.x, 125.0)
        self.assertTrue(self.workspace._pending_canvas_surface_mesh_update)
        self.assertTrue(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        self._assert_four_current_handles(level, wall)

    def test_delayed_failure_after_two_axes_restores_pre_burst_geometry(
        self,
    ) -> None:
        level, wall = self._install_level()
        original_vertices = tuple(level.vertex_data.vertices)
        x_target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        self._finish_changed_drag(x_target, 0.5)
        y_target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_Y,
        )
        self._finish_changed_drag(y_target, 0.4)

        moved_shared = level.vertex_data.get_vertex(2)
        assert moved_shared is not None
        self.assertEqual((moved_shared.x, moved_shared.y), (125.0, -20.0))

        with (
            patch(
                "housemaker.canvas_surface_edits._validate_generated_surfaces",
                side_effect=ValueError("invalid repeated wall edit"),
            ) as validate_geometry,
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
            ) as reconcile_assignments,
        ):
            self.workspace._commit_pending_canvas_surface_mesh_update()

        validate_geometry.assert_called_once()
        reconcile_assignments.assert_not_called()
        self.assertEqual(tuple(level.vertex_data.vertices), original_vertices)
        self.assertFalse(self.workspace._pending_canvas_surface_mesh_update)
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        self._assert_four_current_handles(level, wall)
        restored_x_target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        self.assertEqual(
            restored_x_target.chain_vertex_image_positions[-1],
            (100.0, 0.0),
        )
        assert self.workspace.viewer.window_tools_status_label is not None
        self.assertIn(
            "invalid repeated wall edit",
            self.workspace.viewer.window_tools_status_label.text(),
        )

    def test_selecting_another_surface_flushes_the_pending_wall_edit(
        self,
    ) -> None:
        level, wall = self._install_level()
        target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        self._finish_changed_drag(target, 0.5)
        other_wall = next(
            surface
            for surface in self.workspace._canvas_surface_targets_by_id.values()
            if surface.surface_type == SURFACE_TYPE_WALL
            and surface.surface_id != wall.surface_id
        )

        with (
            patch(
                "housemaker.canvas_surface_edits._validate_generated_surfaces"
            ) as validate_geometry,
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            self.workspace._handle_canvas_surface_selection_changed(
                (other_wall.surface_id,)
            )

        validate_geometry.assert_called_once()
        reconcile_assignments.assert_called_once_with([level])
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertFalse(self.workspace._pending_canvas_surface_mesh_update)
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        self.assertEqual(
            self.workspace._desired_canvas_surface_ids,
            (other_wall.surface_id,),
        )

    def test_accepted_save_flushes_pending_edit_before_serialization(
        self,
    ) -> None:
        level, wall = self._install_level()
        target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        self._finish_changed_drag(target, 0.5)
        saved_state: dict[str, object] = {}

        def capture_save(**kwargs: object) -> None:
            saved_state["pending"] = (
                self.workspace._pending_canvas_surface_mesh_update
            )
            shared = level.vertex_data.get_vertex(2)
            assert shared is not None
            saved_state["shared_x"] = shared.x
            saved_state["levels"] = kwargs["levels"]

        save_path = Path(self._temporary_directory.name) / "project.json"
        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(str(save_path), "JSON Files (*.json)"),
            ),
            patch("housemaker.main.save_project", side_effect=capture_save),
            patch("housemaker.main.QMessageBox.information"),
            patch(
                "housemaker.canvas_surface_edits._validate_generated_surfaces"
            ) as validate_geometry,
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ),
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ),
        ):
            self.workspace._handle_save_clicked()

        validate_geometry.assert_called_once()
        self.assertIs(saved_state["pending"], False)
        self.assertAlmostEqual(float(saved_state["shared_x"]), 125.0)
        self.assertIs(saved_state["levels"], self.workspace.levels)
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )

    def test_cancelled_save_restores_followup_and_restarts_pending_timer(
        self,
    ) -> None:
        level, wall = self._install_level()
        first_target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        self._finish_changed_drag(first_target, 0.5)
        followup_target = self._target(
            wall.surface_id,
            axis_index=CANVAS_SURFACE_EDIT_AXIS_X,
        )
        self.workspace._handle_canvas_surface_edit_started(
            _build_edit(followup_target, 0.0)
        )
        self.workspace._handle_canvas_surface_edit_preview_changed(
            _build_edit(followup_target, 0.2)
        )
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )

        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ),
            patch("housemaker.main.save_project") as save_project,
            patch(
                "housemaker.canvas_surface_edits._validate_generated_surfaces"
            ) as validate_geometry,
        ):
            self.workspace._handle_save_clicked()

        save_project.assert_not_called()
        validate_geometry.assert_not_called()
        shared = level.vertex_data.get_vertex(2)
        assert shared is not None
        self.assertAlmostEqual(shared.x, 125.0)
        self.assertIsNone(self.workspace._active_canvas_surface_edit_target)
        self.assertTrue(self.workspace._pending_canvas_surface_mesh_update)
        self.assertTrue(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        self._assert_four_current_handles(level, wall)


if __name__ == "__main__":
    unittest.main()
