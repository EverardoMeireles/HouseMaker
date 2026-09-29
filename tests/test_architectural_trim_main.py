# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.architectural_trim import (
    TRIM_HANDLE_HEIGHT,
    ArchitecturalTrimPlacementRequest,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import (
    TRIM_KIND_SKIRTING_BOARD,
    LevelData,
    RoomData,
    VertexData,
)
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.viewer import ArchitecturalTrimDimensionEdit

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_square_level() -> LevelData:
    vertex_data = VertexData()
    boundary_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        boundary_ids,
        (*boundary_ids[1:], boundary_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    center = vertex_data.add_vertex(50.0, 50.0)
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[
            RoomData(
                name="Room",
                vertex_ids=boundary_ids,
                center_vertex_id=center.id,
                color_rgb=(140, 180, 220),
            )
        ],
    )


def _wall_surface_id(start_id: int, end_id: int) -> str:
    return f"level:2/room:5/wall:{start_id}:{end_id}"


# ### Main integration tests ###
class ArchitecturalTrimMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.level = _build_square_level()
        self.workspace.levels = [self.level]
        self.workspace.current_level_index = 0
        self.workspace.surface_texture_generation.set_levels([self.level])
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._set_canvas_viewer_targets(build_fixed_surfaces([self.level]))

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _place_skirting(self, trim_id: str, wall_surface_id: str) -> None:
        self.workspace._handle_architectural_trim_placement_requested(
            ArchitecturalTrimPlacementRequest(
                trim_id=trim_id,
                kind=TRIM_KIND_SKIRTING_BOARD,
                wall_surface_ids=(wall_surface_id,),
            )
        )

    def test_placement_selects_semantic_parts_and_undo_removes_component(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("1" * 32, _wall_surface_id(1, 2))

        self.assertEqual(len(self.level.architectural_trims), 1)
        selected_ids = self.workspace._desired_canvas_architectural_trim_part_ids
        self.assertEqual(len(selected_ids), 1)
        self.assertIn("/part:front:wall", selected_ids[0])
        self.assertEqual(
            self.workspace.viewer.get_selected_architectural_trim_part_ids(),
            selected_ids,
        )
        self.assertEqual(
            self.workspace._atlas_surface_assignment_target_ids,
            selected_ids,
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.architectural_trims, [])

    def test_joined_run_dimension_edit_updates_both_members_and_deletes_both(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("2" * 32, _wall_surface_id(1, 2))
            self._place_skirting("3" * 32, _wall_surface_id(2, 3))

        edit = ArchitecturalTrimDimensionEdit(
            trim_id="2" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )
        with patch.object(
            self.workspace,
            "_schedule_viewer_preview_refresh",
        ) as schedule_refresh:
            self.workspace._handle_architectural_trim_edit_started(edit)
            self.workspace._handle_architectural_trim_edit_preview_changed(edit)
            schedule_refresh.assert_not_called()
            self.assertIsNotNone(self.workspace._active_architectural_trim_undo_state)
            self.workspace._handle_architectural_trim_edit_finished(edit, True)
            schedule_refresh.assert_not_called()
            self.assertTrue(
                self.workspace._architectural_trim_mesh_update_timer.isActive()
            )
            self.workspace._commit_pending_architectural_trim_mesh_update()
            schedule_refresh.assert_called_once_with(preserve_camera=True)

        self.assertEqual(
            [trim.height_meters for trim in self.level.architectural_trims],
            [0.18, 0.18],
        )
        self.assertEqual(
            {
                part.run_id
                for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            },
            {"2" * 32},
        )

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_architectural_trim_deletion_requested(
                ("2" * 32, "3" * 32)
            )

        self.assertEqual(self.level.architectural_trims, [])

    def test_dimension_edit_release_keeps_preview_until_delayed_commit(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("5" * 32, _wall_surface_id(1, 2))
            self._place_skirting("6" * 32, _wall_surface_id(2, 3))
        self.workspace._canvas_undo_stack.clear()
        edit = ArchitecturalTrimDimensionEdit(
            trim_id="5" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )

        self.workspace._handle_architectural_trim_edit_started(edit)
        self.workspace._handle_architectural_trim_edit_preview_changed(edit)
        self.assertFalse(
            self.workspace._architectural_trim_mesh_update_timer.isActive()
        )
        preview_parts = tuple(
            part
            for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
            if part.run_id == "5" * 32
        )
        self.workspace.viewer.set_architectural_trim_edit_preview_parts(preview_parts)

        with (
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
            patch.object(
                self.workspace,
                "_reconcile_surface_assignments_with_scene",
                return_value=False,
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_record_canvas_undo_state",
            ) as record_undo,
        ):
            self.workspace._handle_architectural_trim_edit_finished(edit, True)

            self.assertTrue(
                self.workspace._architectural_trim_mesh_update_timer.isActive()
            )
            self.assertIsNotNone(self.workspace._pending_architectural_trim_undo_state)
            self.assertTrue(
                self.workspace.viewer._architectural_trim_edit_preview_parts
            )
            schedule_refresh.assert_not_called()
            reconcile_assignments.assert_not_called()
            record_undo.assert_not_called()

            self.workspace._commit_pending_architectural_trim_mesh_update()

        self.assertFalse(
            self.workspace._architectural_trim_mesh_update_timer.isActive()
        )
        self.assertIsNone(self.workspace._pending_architectural_trim_undo_state)
        self.assertFalse(self.workspace.viewer._architectural_trim_edit_preview_parts)
        reconcile_assignments.assert_called_once()
        record_undo.assert_called_once()
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertEqual(
            [trim.height_meters for trim in self.level.architectural_trims],
            [0.18, 0.18],
        )

    def test_dimension_drag_rebuilds_only_the_affected_run_preview(self) -> None:
        """Dragging must not rebuild or reconcile the complete viewer scene."""

        first_trim_id = "a" * 32
        second_trim_id = "b" * 32
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting(first_trim_id, _wall_surface_id(1, 2))
            self._place_skirting(second_trim_id, _wall_surface_id(3, 4))

        first_run_id = self.workspace._get_architectural_trim_run_id(first_trim_id)
        second_run_id = self.workspace._get_architectural_trim_run_id(second_trim_id)
        self.assertIsNotNone(first_run_id)
        self.assertIsNotNone(second_run_id)
        self.assertNotEqual(first_run_id, second_run_id)
        original_parts = dict(self.workspace._canvas_architectural_trim_parts_by_id)
        original_first_bounds = {
            part.semantic_id: tuple(float(value) for value in part.mesh.bounds.flat)
            for part in original_parts.values()
            if part.run_id == first_run_id
        }
        original_second_bounds = {
            part.semantic_id: tuple(float(value) for value in part.mesh.bounds.flat)
            for part in original_parts.values()
            if part.run_id == second_run_id
        }
        self.workspace.viewer.set_architectural_trim_edit_preview_parts(
            tuple(
                part for part in original_parts.values() if part.run_id == first_run_id
            )
        )
        edit = ArchitecturalTrimDimensionEdit(
            trim_id=first_trim_id,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )

        with (
            patch.object(
                self.workspace,
                "_refresh_canvas_architectural_trim_targets",
            ) as refresh_all_targets,
            patch.object(
                self.workspace,
                "_reconcile_surface_assignments_with_scene",
            ) as reconcile_assignments,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_scene_refresh,
            patch.object(
                self.workspace.viewer,
                "set_architectural_trim_parts",
            ) as replace_all_viewer_parts,
            patch.object(
                self.workspace.viewer,
                "set_architectural_trim_edit_targets",
            ) as replace_all_edit_targets,
            patch(
                "housemaker.main.build_base_fixed_surfaces",
            ) as rebuild_all_base_surfaces,
        ):
            self.workspace._handle_architectural_trim_edit_started(edit)
            self.workspace._handle_architectural_trim_edit_preview_changed(edit)

        refresh_all_targets.assert_not_called()
        reconcile_assignments.assert_not_called()
        schedule_scene_refresh.assert_not_called()
        replace_all_viewer_parts.assert_not_called()
        replace_all_edit_targets.assert_not_called()
        rebuild_all_base_surfaces.assert_not_called()
        self.assertEqual(
            self.workspace._canvas_architectural_trim_parts_by_id,
            original_parts,
        )
        preview_parts = self.workspace.viewer._architectural_trim_edit_preview_parts
        self.assertTrue(preview_parts)
        self.assertEqual({part.run_id for part in preview_parts}, {first_run_id})
        preview_bounds = {
            part.semantic_id: tuple(float(value) for value in part.mesh.bounds.flat)
            for part in preview_parts
        }
        self.assertEqual(preview_bounds.keys(), original_first_bounds.keys())
        self.assertTrue(
            any(
                preview_bounds[semantic_id] != original_first_bounds[semantic_id]
                for semantic_id in preview_bounds
            )
        )
        self.assertEqual(
            {
                part.semantic_id: tuple(float(value) for value in part.mesh.bounds.flat)
                for part in self.workspace._canvas_architectural_trim_parts_by_id.values()
                if part.run_id == second_run_id
            },
            original_second_bounds,
        )

    def test_repeated_dimension_drag_preserves_first_delayed_baseline(
        self,
    ) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("7" * 32, _wall_surface_id(1, 2))
        original_trim = self.level.architectural_trims[0]
        first_edit = ArchitecturalTrimDimensionEdit(
            trim_id="7" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )
        second_edit = ArchitecturalTrimDimensionEdit(
            trim_id="7" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.24,
        )

        self.workspace._handle_architectural_trim_edit_started(first_edit)
        self.workspace._handle_architectural_trim_edit_preview_changed(first_edit)
        self.workspace._handle_architectural_trim_edit_finished(first_edit, True)
        original_baseline = self.workspace._pending_architectural_trim_undo_state
        self.assertIsNotNone(original_baseline)
        self.assertTrue(self.workspace._architectural_trim_mesh_update_timer.isActive())

        self.workspace._handle_architectural_trim_edit_started(second_edit)

        self.assertFalse(
            self.workspace._architectural_trim_mesh_update_timer.isActive()
        )
        self.assertIs(
            self.workspace._pending_architectural_trim_undo_state,
            original_baseline,
        )
        self.workspace._handle_architectural_trim_edit_preview_changed(second_edit)
        self.workspace._handle_architectural_trim_edit_finished(second_edit, True)
        self.assertTrue(self.workspace._architectural_trim_mesh_update_timer.isActive())
        self.assertIs(
            self.workspace._pending_architectural_trim_undo_state,
            original_baseline,
        )
        baseline_trims = dict(original_baseline.architectural_trims_by_level)[
            self.level.index
        ]
        self.assertEqual(baseline_trims, (original_trim,))
        self.assertAlmostEqual(
            self.level.architectural_trims[0].height_meters,
            0.24,
        )

    def test_shared_mesh_delay_restarts_pending_trim_timer(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("8" * 32, _wall_surface_id(1, 2))
        edit = ArchitecturalTrimDimensionEdit(
            trim_id="8" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )
        self.workspace._handle_architectural_trim_edit_started(edit)
        self.workspace._handle_architectural_trim_edit_preview_changed(edit)
        self.workspace._handle_architectural_trim_edit_finished(edit, True)
        self.assertTrue(self.workspace._architectural_trim_mesh_update_timer.isActive())

        self.workspace._set_mesh_edit_update_delay_seconds(2.4)

        timer = self.workspace._architectural_trim_mesh_update_timer
        self.assertEqual(timer.interval(), 2400)
        self.assertTrue(timer.isActive())

    def test_ctrl_z_restores_pending_dimension_edit_before_commit(self) -> None:
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("9" * 32, _wall_surface_id(1, 2))
        original_trim = self.level.architectural_trims[0]
        self.workspace._canvas_undo_stack.clear()
        edit = ArchitecturalTrimDimensionEdit(
            trim_id="9" * 32,
            handle_kind=TRIM_HANDLE_HEIGHT,
            value_meters=0.18,
        )
        self.workspace._handle_architectural_trim_edit_started(edit)
        self.workspace._handle_architectural_trim_edit_preview_changed(edit)
        self.workspace._handle_architectural_trim_edit_finished(edit, True)

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(self.level.architectural_trims, [original_trim])
        self.assertFalse(
            self.workspace._architectural_trim_mesh_update_timer.isActive()
        )
        self.assertIsNone(self.workspace._pending_architectural_trim_undo_state)
        self.assertFalse(self.workspace.viewer._architectural_trim_edit_preview_parts)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_deleting_a_host_wall_removes_its_trim_and_undo_restores_it(
        self,
    ) -> None:
        self.workspace._sync_canvas_to_current_level()
        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self._place_skirting("4" * 32, _wall_surface_id(1, 2))
            self.workspace.canvas.set_selected_vertex_ids((1,))
            self.workspace.canvas._delete_selected_vertices()

        self.assertEqual(self.level.architectural_trims, [])

        with patch.object(self.workspace, "_schedule_viewer_preview_refresh"):
            self.workspace._handle_canvas_undo_requested()

        self.assertEqual(
            tuple(trim.trim_id for trim in self.level.architectural_trims),
            ("4" * 32,),
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
