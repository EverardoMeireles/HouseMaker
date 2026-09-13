# ### Environment setup ###
from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.architectural_surface_edits import SurfaceVertexInsertionRequest
from housemaker.canvas_openings import CanvasOpeningBounds, CanvasOpeningEdit
from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_AXIS_Y,
    CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
    CANVAS_SURFACE_EDIT_WALL_TRANSLATION,
    CanvasSurfaceEdit,
)
from housemaker.generation_state import (
    GeneratedObjectPlacement,
    GeneratedObjectRecord,
    GenerationData,
)
from housemaker.level_coordinates import (
    build_level_base_z_lookup,
    level_image_to_world_xy,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import DoorwayData, LevelData, VertexData, WindowData
from housemaker.surface_geometry import (
    build_fixed_surfaces,
    build_wall_window_placement,
)
from housemaker.surface_texture_state import SurfaceTextureAssignment
from housemaker.texture_atlas_state import TextureAtlasData

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
    ):
        vertex_data.add_edge(start_id, end_id)
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _send_undo_to_viewer(workspace: BlueprintWorkspace) -> None:
    workspace.viewer.view.keyPressEvent(
        QKeyEvent(
            QKeyEvent.Type.KeyPress,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
    )


def _generated_object_record(
    object_id: str,
    placement: GeneratedObjectPlacement | None,
) -> GeneratedObjectRecord:
    """Return one lightweight completed object for placement undo tests."""

    return GeneratedObjectRecord(
        object_id=object_id,
        frame_index=0,
        object_name=f"Object {object_id}",
        pipeline={},
        provider="meshy",
        provider_task_id=f"task-{object_id}",
        asset_path=f"{object_id}.glb",
        placement=placement,
    )


# ### Canvas undo integration tests ###
class CanvasUndoMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        settings = ApplicationSettingsStore(
            Path(self.temporary_directory.name) / "settings.json"
        )
        self.workspace = BlueprintWorkspace(application_settings=settings)
        self.level = _build_square_level()
        self.workspace._apply_project_state([self.level], 0)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _place_wall_vertex(
        self,
        point: tuple[float, float, float],
    ) -> None:
        wall = next(
            surface
            for surface in build_fixed_surfaces([self.level])
            if surface.surface_id == "level:2/wall:1:2"
        )
        self.workspace._handle_canvas_surface_vertex_insertion_requested(
            SurfaceVertexInsertionRequest(
                surface_id=wall.surface_id,
                world_point=point,
                active_vertex_id=(
                    self.workspace._active_canvas_surface_drawing_vertex_id
                ),
            )
        )

    def _wall_translation_target(self):
        """Install and return the bottom wall's delayed Y translation target."""

        surfaces = tuple(build_fixed_surfaces([self.level]))
        self.workspace._set_canvas_viewer_targets(surfaces)
        wall = next(
            surface
            for surface in surfaces
            if surface.surface_id == "level:2/wall:1:2"
        )
        return next(
            candidate
            for candidate in (
                self.workspace._canvas_surface_edit_targets_by_key.values()
            )
            if candidate.surface_id == wall.surface_id
            and candidate.reference.kind
            == CANVAS_SURFACE_EDIT_WALL_TRANSLATION
            and candidate.reference.axis_index == CANVAS_SURFACE_EDIT_AXIS_Y
        )

    def _finish_delayed_surface_edit(
        self,
        target,
        delta_meters: float,
    ) -> None:
        """Apply and release one delayed structural edit without its timer."""

        edit = CanvasSurfaceEdit(
            reference=target.reference,
            surface_id=target.surface_id,
            delta_meters=delta_meters,
        )
        self.workspace._handle_canvas_surface_edit_started(edit)
        self.workspace._handle_canvas_surface_edit_preview_changed(edit)
        self.workspace._handle_canvas_surface_edit_finished(edit, True)

    def test_repeated_ctrl_z_restores_multiple_topology_steps(self) -> None:
        states = [tuple(self.level.editable_surfaces)]
        for point in (
            (0.4, 0.0, 0.8),
            (1.6, 0.0, 0.8),
            (1.6, 0.0, 2.5),
        ):
            self._place_wall_vertex(point)
            states.append(tuple(self.level.editable_surfaces))

        self.assertEqual(len(self.workspace._canvas_undo_stack), 3)
        for expected_state in reversed(states[:-1]):
            _send_undo_to_viewer(self.workspace)
            self.assertEqual(
                tuple(self.level.editable_surfaces),
                expected_state,
            )
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_topology_undo_restores_manual_surface_orientation_flips(self) -> None:
        original_flips = {"level:2/wall:1:2"}
        self.level.flipped_surface_ids = set(original_flips)
        undo_state = self.workspace._capture_canvas_topology_undo_state()
        self.level.flipped_surface_ids = {"level:2/wall:2:3"}
        self.workspace._record_canvas_undo_state(undo_state)

        _send_undo_to_viewer(self.workspace)

        self.assertEqual(self.level.flipped_surface_ids, original_flips)

    def test_deleted_direct_face_is_restored_by_ctrl_z(self) -> None:
        points = (
            (0.4, 0.0, 0.8),
            (1.6, 0.0, 0.8),
            (1.6, 0.0, 2.5),
            (0.4, 0.0, 2.5),
            (0.4, 0.0, 0.8),
        )
        for point in points:
            self._place_wall_vertex(point)
        direct_surface_id = self.workspace._desired_canvas_surface_ids[0]
        before_deletion = tuple(self.level.editable_surfaces)

        self.workspace._handle_canvas_surface_face_deletion_requested(
            (direct_surface_id,)
        )

        self.assertNotIn(
            direct_surface_id,
            {
                surface.surface_id
                for surface in build_fixed_surfaces([self.level])
            },
        )
        _send_undo_to_viewer(self.workspace)
        self.assertEqual(tuple(self.level.editable_surfaces), before_deletion)
        self.assertEqual(
            self.workspace._desired_canvas_surface_ids,
            (direct_surface_id,),
        )

    def test_2d_and_3d_canvas_actions_share_chronological_history(self) -> None:
        initial_topology = tuple(self.level.editable_surfaces)
        self._place_wall_vertex((0.4, 0.0, 0.8))
        topology_after_vertex = tuple(self.level.editable_surfaces)
        vertex_data_before_2d_edit = self.level.vertex_data.clone()

        self.workspace.canvas._push_undo_state()
        self.workspace.canvas.vertex_data.add_vertex(50.0, 50.0)
        self.workspace.canvas.geometry_changed.emit()

        self.assertEqual(len(self.workspace._canvas_undo_stack), 2)
        _send_undo_to_viewer(self.workspace)
        self.assertEqual(self.level.vertex_data, vertex_data_before_2d_edit)
        self.assertEqual(
            tuple(self.level.editable_surfaces),
            topology_after_vertex,
        )

        _send_undo_to_viewer(self.workspace)
        self.assertEqual(tuple(self.level.editable_surfaces), initial_topology)

    def test_face_delete_undo_restores_its_texture_assignment(self) -> None:
        wall_id = "level:2/wall:1:2"
        self.workspace.surface_texture_generation.restore_assignment_snapshot(
            (
                SurfaceTextureAssignment(
                    assignment_id="wall-texture",
                    surface_type="wall",
                    surface_ids=(wall_id,),
                    provider="test",
                    asset_path="wall.png",
                ),
            ),
            emit_signals=False,
        )
        for point in (
            (0.4, 0.0, 0.8),
            (1.6, 0.0, 0.8),
            (1.6, 0.0, 2.5),
            (0.4, 0.0, 2.5),
            (0.4, 0.0, 0.8),
        ):
            self._place_wall_vertex(point)
        direct_surface_id = self.workspace._desired_canvas_surface_ids[0]
        self.assertIn(
            direct_surface_id,
            self.workspace.surface_texture_generation.get_assignments()[
                0
            ].surface_ids,
        )

        self.workspace._handle_canvas_surface_face_deletion_requested(
            (direct_surface_id,)
        )
        self.assertNotIn(
            direct_surface_id,
            self.workspace.surface_texture_generation.get_assignments()[
                0
            ].surface_ids,
        )

        _send_undo_to_viewer(self.workspace)
        self.assertIn(
            direct_surface_id,
            self.workspace.surface_texture_generation.get_assignments()[
                0
            ].surface_ids,
        )

    def test_topology_undo_preserves_a_later_texture_assignment(self) -> None:
        self._place_wall_vertex((0.4, 0.0, 0.8))
        later_assignment = SurfaceTextureAssignment(
            assignment_id="later-texture",
            surface_type="wall",
            surface_ids=("level:2/wall:1:2",),
            provider="test",
            asset_path="later.png",
        )
        self.workspace.surface_texture_generation.restore_assignment_snapshot(
            (later_assignment,),
            emit_signals=False,
        )

        _send_undo_to_viewer(self.workspace)

        self.assertEqual(
            self.workspace.surface_texture_generation.get_assignments(),
            (later_assignment,),
        )

    def test_topology_undo_keeps_newer_targets_for_the_same_assignment(
        self,
    ) -> None:
        wall_id = "level:2/wall:1:2"
        original_assignment = SurfaceTextureAssignment(
            assignment_id="shared-texture",
            surface_type="wall",
            surface_ids=(wall_id,),
            provider="test",
            asset_path="wall.png",
        )
        self.workspace.surface_texture_generation.restore_assignment_snapshot(
            (original_assignment,),
            emit_signals=False,
        )
        for point in (
            (0.4, 0.0, 0.8),
            (1.6, 0.0, 0.8),
            (1.6, 0.0, 2.5),
            (0.4, 0.0, 2.5),
            (0.4, 0.0, 0.8),
        ):
            self._place_wall_vertex(point)
        other_wall_id = next(
            surface.surface_id
            for surface in build_fixed_surfaces([self.level])
            if surface.surface_type == "wall"
            and surface.source_surface_id is None
            and surface.surface_id != wall_id
        )
        reassigned = replace(
            self.workspace.surface_texture_generation.get_assignments()[0],
            surface_ids=(other_wall_id,),
            combined_area_m2=1.0,
            area_description="newer selection",
        )
        self.workspace.surface_texture_generation.restore_assignment_snapshot(
            (reassigned,),
            emit_signals=False,
        )

        _send_undo_to_viewer(self.workspace)

        self.assertEqual(
            self.workspace.surface_texture_generation.get_assignments(),
            (reassigned,),
        )
        assert self.workspace.viewer.surface_tools_status_label is not None
        self.assertIn(
            "newer texture bindings were kept",
            self.workspace.viewer.surface_tools_status_label.text(),
        )

    def test_ctrl_z_restores_a_committed_delayed_wall_edit(self) -> None:
        surfaces = tuple(build_fixed_surfaces([self.level]))
        self.workspace._set_canvas_viewer_targets(surfaces)
        wall = next(
            surface
            for surface in surfaces
            if surface.surface_id == "level:2/wall:1:2"
        )
        target = next(
            candidate
            for candidate in (
                self.workspace._canvas_surface_edit_targets_by_key.values()
            )
            if candidate.surface_id == wall.surface_id
            and candidate.reference.kind
            == CANVAS_SURFACE_EDIT_WALL_TRANSLATION
            and candidate.reference.axis_index == CANVAS_SURFACE_EDIT_AXIS_Y
        )
        original_vertices = self.level.vertex_data.clone()
        edit = CanvasSurfaceEdit(
            reference=target.reference,
            surface_id=target.surface_id,
            delta_meters=0.2,
        )

        self.workspace._handle_canvas_surface_edit_started(edit)
        self.workspace._handle_canvas_surface_edit_preview_changed(edit)
        self.workspace._handle_canvas_surface_edit_finished(edit, True)
        self.assertTrue(
            self.workspace._pending_canvas_surface_mesh_update
        )
        self.workspace._commit_pending_canvas_surface_mesh_update()

        self.assertNotEqual(self.level.vertex_data, original_vertices)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

        _send_undo_to_viewer(self.workspace)

        self.assertEqual(self.level.vertex_data, original_vertices)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_ctrl_z_restores_a_window_gizmo_edit(self) -> None:
        wall = next(
            surface
            for surface in build_fixed_surfaces([self.level])
            if surface.surface_id == "level:2/wall:1:2"
        )
        original_window = WindowData(
            window_id="editable-window",
            wall_surface_id=wall.surface_id,
            start_ratio=0.2,
            end_ratio=0.4,
            bottom_ratio=0.25,
            top_ratio=0.7,
        )
        self.level.windows.append(original_window)
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._set_canvas_viewer_targets(
            tuple(build_fixed_surfaces([self.level]))
        )
        target = next(
            candidate
            for candidate in self.workspace._canvas_opening_targets_by_key.values()
            if candidate.reference.stable_id == original_window.window_id
        )
        starting_edit = CanvasOpeningEdit(
            reference=target.reference,
            wall_surface_id=target.wall_surface_id,
            bounds=target.bounds,
        )
        changed_edit = CanvasOpeningEdit(
            reference=target.reference,
            wall_surface_id=target.wall_surface_id,
            bounds=CanvasOpeningBounds(0.3, 0.55, 0.2, 0.8),
        )

        self.workspace._handle_canvas_opening_edit_started(starting_edit)
        self.workspace._handle_canvas_opening_edit_preview_changed(changed_edit)
        self.workspace._handle_canvas_opening_edit_finished(changed_edit, True)

        self.assertNotEqual(self.level.windows[0], original_window)
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

        _send_undo_to_viewer(self.workspace)

        self.assertEqual(self.level.windows[0], original_window)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_ctrl_z_restores_object_removal_then_its_previous_transform(
        self,
    ) -> None:
        original_placement = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=20.0,
            image_y=30.0,
            height_offset_meters=0.4,
            rotation_degrees=(0.0, 0.0, 0.0),
        )
        self.workspace.generation.set_data(
            GenerationData(
                generated_objects=[
                    _generated_object_record("chair", original_placement)
                ]
            )
        )
        image_x = 75.0
        image_y = 65.0
        world_x, world_y = level_image_to_world_xy(
            self.level,
            image_x,
            image_y,
        )
        base_z = build_level_base_z_lookup((self.level,))[self.level.index]
        transformed_placement = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=image_x,
            image_y=image_y,
            height_offset_meters=1.25,
            rotation_degrees=(10.0, 20.0, 30.0),
        )

        self.workspace._handle_placed_object_transform_changed(
            "chair",
            (world_x, world_y, base_z + 1.25),
            transformed_placement.rotation_degrees,
        )
        self.assertEqual(
            self.workspace.generation.get_generated_object_placement("chair"),
            transformed_placement,
        )
        self.workspace._handle_placed_object_removal_requested("chair")
        self.assertIsNone(
            self.workspace.generation.get_generated_object_placement("chair")
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), 2)

        _send_undo_to_viewer(self.workspace)
        self.assertEqual(
            self.workspace.generation.get_generated_object_placement("chair"),
            transformed_placement,
        )

        _send_undo_to_viewer(self.workspace)
        self.assertEqual(
            self.workspace.generation.get_generated_object_placement("chair"),
            original_placement,
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_transform_undo_restores_the_multi_object_selection(self) -> None:
        chair_placement = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=20.0,
            image_y=30.0,
        )
        table_placement = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=40.0,
            image_y=50.0,
        )
        self.workspace.generation.set_data(
            GenerationData(
                generated_objects=[
                    _generated_object_record("chair", chair_placement),
                    _generated_object_record("table", table_placement),
                ]
            )
        )
        self.workspace._desired_canvas_object_ids = ("chair", "table")
        self.workspace._desired_canvas_object_id = "table"
        world_x, world_y = level_image_to_world_xy(
            self.level,
            70.0,
            60.0,
        )
        base_z = build_level_base_z_lookup((self.level,))[self.level.index]

        self.workspace._handle_placed_object_transform_changed(
            "table",
            (world_x, world_y, base_z),
            (0.0, 0.0, 20.0),
        )
        _send_undo_to_viewer(self.workspace)

        self.assertEqual(
            self.workspace._desired_canvas_object_ids,
            ("chair", "table"),
        )
        self.assertEqual(self.workspace._desired_canvas_object_id, "table")

    def test_ctrl_z_removes_a_newly_added_window(self) -> None:
        wall = next(
            surface
            for surface in build_fixed_surfaces([self.level])
            if surface.surface_id == "level:2/wall:1:2"
        )
        placement = build_wall_window_placement(
            wall,
            (0.5, 0.0, 0.75),
            (1.5, 0.0, 2.25),
        )

        with (
            patch.object(
                self.workspace,
                "_build_model_with_stable_dependencies",
                return_value=(object(), ("window-undo",)),
            ),
            patch.object(
                self.workspace,
                "_apply_canvas_window_preview",
                return_value=True,
            ),
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)
            self.assertEqual(len(self.level.windows), 1)
            self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

            _send_undo_to_viewer(self.workspace)

        self.assertEqual(self.level.windows, [])
        self.assertEqual(self.workspace._canvas_window_undo_ids, [])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_newer_2d_edit_is_undone_before_a_pending_wall_edit(self) -> None:
        target = self._wall_translation_target()
        original_vertices = self.level.vertex_data.clone()
        self._finish_delayed_surface_edit(target, 0.2)
        wall_edited_vertices = self.level.vertex_data.clone()
        self.assertTrue(
            self.workspace._pending_canvas_surface_mesh_update
        )

        self.workspace.canvas._push_undo_state()
        self.workspace.canvas.vertex_data.add_vertex(250.0, 250.0)
        self.workspace.canvas.geometry_changed.emit()

        _send_undo_to_viewer(self.workspace)

        self.assertEqual(
            self.level.vertex_data,
            wall_edited_vertices,
            "The newer 2D edit must be undone before the pending wall edit.",
        )

        _send_undo_to_viewer(self.workspace)
        self.assertEqual(self.level.vertex_data, original_vertices)

    def test_dedicated_window_undo_discards_later_edits_for_that_window(
        self,
    ) -> None:
        original_vertices = self.level.vertex_data.clone()
        self.workspace.canvas._push_undo_state()
        self.workspace.canvas.vertex_data.add_vertex(250.0, 250.0)
        wall = next(
            surface
            for surface in build_fixed_surfaces([self.level])
            if surface.surface_id == "level:2/wall:1:2"
        )
        placement = build_wall_window_placement(
            wall,
            (0.5, 0.0, 0.75),
            (1.5, 0.0, 2.25),
        )

        with (
            patch.object(
                self.workspace,
                "_build_model_with_stable_dependencies",
                return_value=(object(), ("window-history",)),
            ),
            patch.object(
                self.workspace,
                "_apply_canvas_window_preview",
                return_value=True,
            ),
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)
            added_window = self.level.windows[0]
            self.workspace._set_canvas_viewer_targets(
                tuple(build_fixed_surfaces([self.level]))
            )
            target = next(
                candidate
                for candidate in (
                    self.workspace._canvas_opening_targets_by_key.values()
                )
                if candidate.reference.stable_id == added_window.window_id
            )
            starting_edit = CanvasOpeningEdit(
                reference=target.reference,
                wall_surface_id=target.wall_surface_id,
                bounds=target.bounds,
            )
            changed_edit = CanvasOpeningEdit(
                reference=target.reference,
                wall_surface_id=target.wall_surface_id,
                bounds=CanvasOpeningBounds(0.3, 0.6, 0.2, 0.8),
            )
            self.workspace._handle_canvas_opening_edit_started(starting_edit)
            self.workspace._handle_canvas_opening_edit_preview_changed(
                changed_edit
            )
            self.workspace._handle_canvas_opening_edit_finished(
                changed_edit,
                True,
            )

            self.workspace._handle_canvas_window_undo_requested()
            self.assertEqual(self.level.windows, [])

            _send_undo_to_viewer(self.workspace)

        self.assertEqual(
            self.level.vertex_data,
            original_vertices,
            "A removed window's stale edit must not block earlier history.",
        )
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_level_transform_and_height_changes_precede_wall_undo(self) -> None:
        target = self._wall_translation_target()
        original_vertices = self.level.vertex_data.clone()
        original_scale = self.level.scale
        original_x_offset = self.level.offset_x_meters
        original_y_offset = self.level.offset_y_meters
        original_height = self.level.height_meters
        self._finish_delayed_surface_edit(target, 0.2)
        wall_edited_vertices = self.level.vertex_data.clone()
        self.assertTrue(self.workspace._pending_canvas_surface_mesh_update)

        self.workspace._handle_level_scale_changed(original_scale + 0.5)
        self.assertFalse(self.workspace._pending_canvas_surface_mesh_update)
        self.workspace._handle_level_x_offset_changed(original_x_offset + 2.0)
        self.workspace._handle_level_y_offset_changed(original_y_offset - 3.0)
        self.workspace._handle_height_level_changed(original_height + 1.0)

        _send_undo_to_viewer(self.workspace)
        self.assertEqual(self.level.height_meters, original_height)
        self.assertEqual(self.level.vertex_data, wall_edited_vertices)

        _send_undo_to_viewer(self.workspace)
        self.assertEqual(self.level.offset_x_meters, original_x_offset)
        self.assertEqual(self.level.offset_y_meters, original_y_offset)
        self.assertEqual(self.level.scale, original_scale)
        self.assertEqual(self.level.vertex_data, wall_edited_vertices)

        _send_undo_to_viewer(self.workspace)
        self.assertEqual(self.level.vertex_data, original_vertices)

    def test_2d_doorway_drag_returned_to_start_leaves_no_shared_undo(
        self,
    ) -> None:
        doorway = DoorwayData(
            center_x=50.0,
            center_y=0.0,
            width_meters=1.0,
            height_meters=2.0,
            rotation_degrees=90.0,
        )
        self.level.doorways.append(doorway)
        self.workspace._sync_canvas_to_current_level()
        canvas = self.workspace.canvas
        canvas.pressed_doorway_index = 0
        canvas.drag_doorway_index = 0
        canvas.doorway_drag_press_image_point = (50.0, 0.0)
        canvas.doorway_drag_initial_doorway = copy.deepcopy(doorway)
        canvas.doorway_drag_wall_edge = canvas.vertex_data.edges[0]

        canvas._move_dragged_doorway(QPointF(65.0, 0.0))
        canvas._move_dragged_doorway(QPointF(50.0, 0.0))
        canvas._reset_doorway_pointer_state()

        self.assertEqual(canvas.doorways[0], doorway)
        self.assertEqual(canvas.undo_stack, [])
        self.assertEqual(
            self.workspace._canvas_undo_stack,
            [],
            "The shared history must discard the reverted drag snapshot too.",
        )

    def test_wall_change_away_and_back_does_not_add_undo(self) -> None:
        target = self._wall_translation_target()
        original_vertices = self.level.vertex_data.clone()
        start_edit = CanvasSurfaceEdit(
            reference=target.reference,
            surface_id=target.surface_id,
            delta_meters=0.0,
        )
        changed_edit = replace(start_edit, delta_meters=0.2)

        self.workspace._handle_canvas_surface_edit_started(start_edit)
        self.workspace._handle_canvas_surface_edit_preview_changed(changed_edit)
        self.workspace._handle_canvas_surface_edit_preview_changed(start_edit)
        self.workspace._handle_canvas_surface_edit_finished(start_edit, True)
        self.workspace._commit_pending_canvas_surface_mesh_update()

        self.assertEqual(self.level.vertex_data, original_vertices)
        self.assertEqual(
            self.workspace._canvas_undo_stack,
            [],
            "A wall edit ending at its baseline must not enter history.",
        )

    def test_floor_change_away_and_back_does_not_add_undo(self) -> None:
        surfaces = tuple(build_fixed_surfaces([self.level]))
        self.workspace._set_canvas_viewer_targets(surfaces)
        target = next(
            candidate
            for candidate in (
                self.workspace._canvas_surface_edit_targets_by_key.values()
            )
            if candidate.reference.kind
            == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS
        )
        original_thickness = self.level.floor_thickness_meters
        start_edit = CanvasSurfaceEdit(
            reference=target.reference,
            surface_id=target.surface_id,
            delta_meters=0.0,
        )
        changed_edit = replace(start_edit, delta_meters=0.1)

        self.workspace._handle_canvas_surface_edit_started(start_edit)
        self.workspace._handle_canvas_surface_edit_preview_changed(changed_edit)
        self.workspace._handle_canvas_surface_edit_preview_changed(start_edit)
        self.workspace._handle_canvas_surface_edit_finished(start_edit, True)
        self.workspace._commit_pending_canvas_surface_mesh_update()

        self.assertEqual(
            self.level.floor_thickness_meters,
            original_thickness,
        )
        self.assertEqual(
            self.workspace._canvas_undo_stack,
            [],
            "A floor edit ending at its baseline must not enter history.",
        )

    def test_ctrl_z_removes_an_initial_completed_object_placement(self) -> None:
        self.workspace.generation.set_data(
            GenerationData(
                generated_objects=[_generated_object_record("chair", None)]
            )
        )
        self.assertTrue(
            self.workspace.generation.request_generated_object_placement(
                "chair"
            )
        )
        dialog = self.workspace._object_placement_dialog
        request_id = self.workspace._object_placement_operation_id
        self.assertIsNotNone(dialog)
        self.assertIsNotNone(request_id)
        placement = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=25.0,
            image_y=35.0,
        )

        self.workspace._handle_object_placement_selected(
            dialog,  # type: ignore[arg-type]
            request_id,  # type: ignore[arg-type]
            placement,
        )
        self.assertEqual(
            self.workspace.generation.get_generated_object_placement("chair"),
            placement,
        )

        _send_undo_to_viewer(self.workspace)

        self.assertIsNone(
            self.workspace.generation.get_generated_object_placement("chair")
        )

    def test_ctrl_z_restores_a_completed_object_replacement(self) -> None:
        original = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=20.0,
            image_y=30.0,
            height_offset_meters=0.75,
            rotation_degrees=(5.0, 10.0, 15.0),
        )
        self.workspace.generation.set_data(
            GenerationData(
                generated_objects=[
                    _generated_object_record("chair", original)
                ]
            )
        )
        self.assertTrue(
            self.workspace.generation.request_generated_object_placement(
                "chair"
            )
        )
        dialog = self.workspace._object_placement_dialog
        request_id = self.workspace._object_placement_operation_id
        self.assertIsNotNone(dialog)
        self.assertIsNotNone(request_id)

        self.workspace._handle_object_placement_selected(
            dialog,  # type: ignore[arg-type]
            request_id,  # type: ignore[arg-type]
            GeneratedObjectPlacement(
                level_index=self.level.index,
                image_x=80.0,
                image_y=70.0,
            ),
        )
        self.assertNotEqual(
            self.workspace.generation.get_generated_object_placement("chair"),
            original,
        )

        _send_undo_to_viewer(self.workspace)

        self.assertEqual(
            self.workspace.generation.get_generated_object_placement("chair"),
            original,
        )

    def test_async_completion_records_its_pending_initial_placement(self) -> None:
        placement = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=45.0,
            image_y=55.0,
        )
        record = _generated_object_record("async-chair", placement)
        self.workspace.generation.set_data(
            GenerationData(generated_objects=[record])
        )

        self.workspace._handle_generated_object_completed_for_canvas(
            record,
            object(),
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

        _send_undo_to_viewer(self.workspace)

        self.assertIsNone(
            self.workspace.generation.get_generated_object_placement(
                "async-chair"
            )
        )

    def test_atlas_remove_records_and_restores_a_placed_object(self) -> None:
        placement = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=40.0,
            image_y=60.0,
        )
        self.workspace.generation.set_data(
            GenerationData(
                generated_objects=[
                    _generated_object_record("chair", placement)
                ]
            )
        )
        atlas_data = TextureAtlasData()
        atlas_data.create_atlas("Objects", 2048, atlas_id="objects")
        atlas_data.assign_object(
            "objects",
            "chair",
            "chair.png",
            512,
        )
        self.workspace.texture_atlas_workspace.set_data(atlas_data)

        with (
            patch.object(
                self.workspace,
                "_sync_atlas_object_texture_sources",
            ),
            patch.object(
                self.workspace.texture_atlas_workspace,
                "refresh_texture_source_content",
            ) as refresh_texture_source_content,
        ):
            self.workspace._handle_atlas_source_remove_requested(
                "object",
                "chair",
            )
            self.assertIsNone(
                self.workspace.generation.get_generated_object_placement(
                    "chair"
                )
            )
            _send_undo_to_viewer(self.workspace)
            refresh_texture_source_content.assert_any_call(("chair",))

        self.assertEqual(
            self.workspace.generation.get_generated_object_placement("chair"),
            placement,
        )
        restored_atlas = (
            self.workspace.texture_atlas_workspace.get_data().atlas_by_id(
                "objects"
            )
        )
        self.assertIsNotNone(restored_atlas)
        assert restored_atlas is not None
        self.assertIsNotNone(restored_atlas.placement_for_object("chair"))

    def test_atlas_restore_keeps_a_newer_global_placement_and_reports_it(
        self,
    ) -> None:
        placement = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=40.0,
            image_y=60.0,
        )
        self.workspace.generation.set_data(
            GenerationData(
                generated_objects=[
                    _generated_object_record("chair", placement)
                ]
            )
        )
        atlas_data = TextureAtlasData()
        atlas_data.create_atlas("First", 2048, atlas_id="first")
        atlas_data.create_atlas("Second", 2048, atlas_id="second")
        atlas_data.assign_object("first", "chair", "chair.png", 512)
        self.workspace.texture_atlas_workspace.set_data(atlas_data)

        with patch.object(
            self.workspace,
            "_sync_atlas_object_texture_sources",
        ):
            self.workspace._handle_atlas_source_remove_requested(
                "object",
                "chair",
            )
            newer_data = self.workspace.texture_atlas_workspace.get_data()
            newer_data.assign_object(
                "second",
                "chair",
                "newer-chair.png",
                512,
            )
            self.workspace.texture_atlas_workspace.set_data(newer_data)
            _send_undo_to_viewer(self.workspace)

        current_data = self.workspace.texture_atlas_workspace.get_data()
        first = current_data.atlas_by_id("first")
        second = current_data.atlas_by_id("second")
        assert first is not None and second is not None
        self.assertIsNone(first.placement_for_object("chair"))
        self.assertIsNotNone(second.placement_for_object("chair"))
        self.assertIn(
            "could not be restored",
            self.workspace.viewer.surface_tools_status_label.text(),
        )

    def test_ctrl_z_from_canvas_side_panel_uses_shared_history(self) -> None:
        original_height = self.level.height_meters
        self.workspace.height_level_spinbox.setValue(original_height + 1.0)
        self.assertEqual(self.level.height_meters, original_height + 1.0)
        self.workspace.show()
        self.workspace.save_button.setFocus()
        _qt_application.processEvents()

        QTest.keyClick(
            self.workspace.save_button,
            Qt.Key.Key_Z,
            Qt.KeyboardModifier.ControlModifier,
        )
        _qt_application.processEvents()

        self.assertEqual(self.level.height_meters, original_height)
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_replacing_blueprint_image_starts_a_new_undo_history(self) -> None:
        self._place_wall_vertex((0.4, 0.0, 0.8))
        self.assertTrue(self.workspace._canvas_undo_stack)
        image_path = Path(self.temporary_directory.name) / "replacement.png"
        Image.new("RGB", (64, 32), "white").save(image_path)

        self.workspace._set_current_level_image(str(image_path))

        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.assertEqual(self.workspace.canvas.undo_stack, [])
        self.assertEqual(self.level.image_size_pixels, (64.0, 32.0))

    def test_object_deletion_prunes_stale_placement_history(self) -> None:
        initial_topology = tuple(self.level.editable_surfaces)
        self._place_wall_vertex((0.4, 0.0, 0.8))
        placement = GeneratedObjectPlacement(
            level_index=self.level.index,
            image_x=20.0,
            image_y=30.0,
        )
        self.workspace.generation.set_data(
            GenerationData(
                generated_objects=[
                    _generated_object_record("chair", placement)
                ]
            )
        )
        world_x, world_y = level_image_to_world_xy(
            self.level,
            70.0,
            60.0,
        )
        base_z = build_level_base_z_lookup((self.level,))[self.level.index]
        self.workspace._handle_placed_object_transform_changed(
            "chair",
            (world_x, world_y, base_z),
            (0.0, 0.0, 20.0),
        )

        self.workspace._handle_generated_object_deleted_for_canvas("chair")
        _send_undo_to_viewer(self.workspace)

        self.assertEqual(tuple(self.level.editable_surfaces), initial_topology)
        self.assertEqual(self.workspace._canvas_undo_stack, [])


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
