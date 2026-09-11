# ### Imports ###
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from housemaker.architectural_surface_edits import (
    insert_surface_vertex,
    place_surface_vertex,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData
from housemaker.surface_geometry import build_fixed_surfaces
from housemaker.texture_atlas_state import TextureAtlasData


# ### Fixture helpers ###
def _build_square_level() -> LevelData:
    vertex_data = VertexData()
    vertex_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for start_id, end_id in zip(
        vertex_ids,
        (*vertex_ids[1:], vertex_ids[0]),
        strict=True,
    ):
        vertex_data.add_edge(start_id, end_id)
    return LevelData(index=2, name="Ground", vertex_data=vertex_data)


def _build_workspace(level: LevelData) -> SimpleNamespace:
    surface_textures = Mock()
    surface_textures.remap_assignments_with_surface_lineage.return_value = True
    surface_textures.reconcile_assignments_with_levels.return_value = False
    surface_textures.snapshot_assignments.return_value = ()
    texture_atlases = Mock()
    texture_atlases.get_data.return_value = TextureAtlasData()
    return SimpleNamespace(
        levels=[level],
        _desired_canvas_surface_ids=("old-surface",),
        _atlas_surface_assignment_target_ids=("old-surface",),
        _desired_canvas_object_id="old-object",
        _active_canvas_surface_drawing_vertex_id=None,
        _canvas_undo_stack=[],
        _is_restoring_canvas_undo=False,
        _record_canvas_undo_state=Mock(),
        _commit_pending_canvas_surface_mesh_update=Mock(),
        _commit_pending_wall_vertex_update=Mock(),
        _commit_pending_doorway_mesh_update=Mock(),
        _canvas_surface_mesh_update_timer=Mock(),
        _wall_vertex_update_timer=Mock(),
        _doorway_mesh_update_timer=Mock(),
        surface_texture_generation=surface_textures,
        texture_atlas_workspace=texture_atlases,
        viewer=Mock(),
        _sync_canvas_surface_drawing_overlay=Mock(),
        _schedule_viewer_preview_refresh=Mock(),
    )


def _root_floor_and_point(level: LevelData) -> tuple[str, tuple[float, ...]]:
    surface = next(
        candidate
        for candidate in build_fixed_surfaces((level,))
        if candidate.surface_type == "floor"
    )
    point = tuple(float(value) for value in surface.mesh.triangles[0].mean(axis=0))
    return surface.surface_id, point


def _root_floor_and_inner_loop(
    level: LevelData,
) -> tuple[str, tuple[tuple[float, ...], ...]]:
    surface = next(
        candidate
        for candidate in build_fixed_surfaces((level,))
        if candidate.surface_type == "floor"
    )
    first, second, third = surface.mesh.triangles[0]
    points = tuple(
        tuple(float(value) for value in point)
        for point in (
            (0.6 * first) + (0.2 * second) + (0.2 * third),
            (0.2 * first) + (0.6 * second) + (0.2 * third),
            (0.2 * first) + (0.2 * second) + (0.6 * third),
        )
    )
    return surface.surface_id, points


# ### Main topology transaction tests ###
class CanvasSurfaceTopologyMainTests(unittest.TestCase):
    def test_loop_closure_remaps_texture_once_and_refreshes_mesh_once(self) -> None:
        level = _build_square_level()
        workspace = _build_workspace(level)
        source_id, points = _root_floor_and_inner_loop(level)
        results = []

        for point in (*points, points[0]):
            active_vertex_id = workspace._active_canvas_surface_drawing_vertex_id

            def place_vertex(
                point=point,
                active_vertex_id=active_vertex_id,
            ):
                result = place_surface_vertex(
                    (level,),
                    source_id,
                    point,
                    active_vertex_id,
                )
                results.append(result)
                return result

            BlueprintWorkspace._apply_canvas_surface_topology_edit(
                workspace,
                place_vertex,
                success_message="done",
            )

        closure = results[-1]
        self.assertTrue(closure.created_surface_ids)
        self.assertTrue(closure.requires_mesh_refresh)
        remap = (
            workspace.surface_texture_generation.remap_assignments_with_surface_lineage
        )
        remap.assert_called_once_with([level], closure.replacements)
        workspace.surface_texture_generation.reconcile_assignments_with_levels.assert_called_once_with(
            [level]
        )
        workspace._schedule_viewer_preview_refresh.assert_called_once_with(
            preserve_camera=True
        )
        self.assertEqual(
            workspace._desired_canvas_surface_ids,
            closure.created_surface_ids,
        )

    def test_edge_to_edge_cut_remaps_and_refreshes_only_when_partitioned(
        self,
    ) -> None:
        level = _build_square_level()
        workspace = _build_workspace(level)
        floor = next(
            surface
            for surface in build_fixed_surfaces((level,))
            if surface.surface_type == "floor"
        )
        height = float(floor.mesh.bounds[0, 2])
        results = []

        for point in ((0.0, -1.0, height), (2.0, -1.0, height)):
            active_vertex_id = workspace._active_canvas_surface_drawing_vertex_id

            def place_vertex(
                point=point,
                active_vertex_id=active_vertex_id,
            ):
                result = place_surface_vertex(
                    (level,),
                    floor.surface_id,
                    point,
                    active_vertex_id,
                )
                results.append(result)
                return result

            BlueprintWorkspace._apply_canvas_surface_topology_edit(
                workspace,
                place_vertex,
                success_message="done",
            )

        first, partition = results
        self.assertFalse(first.requires_mesh_refresh)
        self.assertTrue(partition.requires_mesh_refresh)
        self.assertEqual(partition.created_surface_ids, ())
        self.assertEqual(len(partition.selected_surface_ids), 2)
        self.assertEqual(
            workspace._desired_canvas_surface_ids,
            partition.selected_surface_ids,
        )
        self.assertEqual(
            workspace._atlas_surface_assignment_target_ids,
            partition.selected_surface_ids,
        )
        self.assertEqual(
            workspace._active_canvas_surface_drawing_vertex_id,
            partition.active_vertex_id,
        )
        remap = (
            workspace.surface_texture_generation
            .remap_assignments_with_surface_lineage
        )
        remap.assert_called_once_with(
            [level],
            partition.replacements,
        )
        reconcile = (
            workspace.surface_texture_generation
            .reconcile_assignments_with_levels
        )
        reconcile.assert_called_once_with([level])
        workspace._schedule_viewer_preview_refresh.assert_called_once_with(
            preserve_camera=True
        )
        workspace._canvas_surface_mesh_update_timer.start.assert_not_called()
        workspace._wall_vertex_update_timer.start.assert_not_called()
        workspace._doorway_mesh_update_timer.start.assert_not_called()

    def test_open_chain_updates_overlay_without_texture_or_mesh_refresh(self) -> None:
        level = _build_square_level()
        workspace = _build_workspace(level)
        surface_id, point = _root_floor_and_point(level)
        result_holder = []

        def place_vertex():
            result = place_surface_vertex((level,), surface_id, point)
            result_holder.append(result)
            return result

        BlueprintWorkspace._apply_canvas_surface_topology_edit(
            workspace,
            place_vertex,
            success_message="done",
        )

        self.assertEqual(len(result_holder), 1)
        result = result_holder[0]
        self.assertFalse(result.requires_mesh_refresh)
        self.assertTrue(result.state_changed)
        self.assertIsNotNone(result.active_vertex_id)
        self.assertEqual(
            workspace._active_canvas_surface_drawing_vertex_id,
            result.active_vertex_id,
        )
        self.assertFalse(level.editable_surfaces[0].replaces_source_surface)
        workspace.surface_texture_generation.remap_assignments_with_surface_lineage.assert_not_called()
        workspace.surface_texture_generation.reconcile_assignments_with_levels.assert_not_called()
        workspace._sync_canvas_surface_drawing_overlay.assert_called_once_with()
        workspace._schedule_viewer_preview_refresh.assert_not_called()

    def test_success_remaps_texture_lineage_before_refresh(self) -> None:
        level = _build_square_level()
        workspace = _build_workspace(level)
        surface_id, point = _root_floor_and_point(level)

        BlueprintWorkspace._apply_canvas_surface_topology_edit(
            workspace,
            lambda: insert_surface_vertex((level,), surface_id, point),
            success_message="done",
        )

        self.assertEqual(len(level.editable_surfaces), 1)
        remap_call = (
            workspace.surface_texture_generation
            .remap_assignments_with_surface_lineage.call_args
        )
        self.assertEqual(remap_call.args[0], [level])
        self.assertIn(surface_id, remap_call.args[1])
        self.assertTrue(workspace._desired_canvas_surface_ids)
        self.assertEqual(
            workspace._desired_canvas_surface_ids,
            workspace._atlas_surface_assignment_target_ids,
        )
        self.assertIsNone(workspace._desired_canvas_object_id)
        workspace.viewer.set_surface_tools_status.assert_called_once_with("done")
        workspace._schedule_viewer_preview_refresh.assert_called_with(
            preserve_camera=True
        )

    def test_remap_failure_rolls_back_geometry_and_selection(self) -> None:
        level = _build_square_level()
        workspace = _build_workspace(level)
        remap = (
            workspace.surface_texture_generation
            .remap_assignments_with_surface_lineage
        )
        remap.side_effect = RuntimeError("lineage failed")
        surface_id, point = _root_floor_and_point(level)

        BlueprintWorkspace._apply_canvas_surface_topology_edit(
            workspace,
            lambda: insert_surface_vertex((level,), surface_id, point),
            success_message="done",
        )

        self.assertEqual(level.editable_surfaces, [])
        self.assertEqual(workspace._desired_canvas_surface_ids, ("old-surface",))
        self.assertEqual(
            workspace._atlas_surface_assignment_target_ids,
            ("old-surface",),
        )
        self.assertEqual(workspace._desired_canvas_object_id, "old-object")
        workspace.viewer.set_surface_tools_status.assert_called_once_with(
            "Surface edit stopped: lineage failed"
        )
        workspace.surface_texture_generation.reconcile_assignments_with_levels.assert_not_called()

    def test_remap_failure_restores_active_chain_endpoint_and_overlay(self) -> None:
        level = _build_square_level()
        workspace = _build_workspace(level)
        previous_active_vertex_id = "a" * 32
        workspace._active_canvas_surface_drawing_vertex_id = previous_active_vertex_id
        remap = (
            workspace.surface_texture_generation
            .remap_assignments_with_surface_lineage
        )
        remap.side_effect = RuntimeError("lineage failed")
        surface_id, point = _root_floor_and_point(level)

        BlueprintWorkspace._apply_canvas_surface_topology_edit(
            workspace,
            lambda: insert_surface_vertex((level,), surface_id, point),
            success_message="done",
        )

        self.assertEqual(
            workspace._active_canvas_surface_drawing_vertex_id,
            previous_active_vertex_id,
        )
        workspace._sync_canvas_surface_drawing_overlay.assert_called_once_with()

    def test_rejected_operation_does_not_rebuild_an_unchanged_model(self) -> None:
        level = _build_square_level()
        workspace = _build_workspace(level)

        BlueprintWorkspace._apply_canvas_surface_topology_edit(
            workspace,
            Mock(side_effect=ValueError("degenerate source face")),
            success_message="done",
        )

        self.assertEqual(level.editable_surfaces, [])
        workspace._sync_canvas_surface_drawing_overlay.assert_called_once_with()
        workspace.viewer.set_surface_tools_status.assert_called_once_with(
            "Surface edit stopped: degenerate source face"
        )
        workspace._schedule_viewer_preview_refresh.assert_not_called()

    def test_chain_reset_clears_only_the_transient_active_endpoint(self) -> None:
        workspace = _build_workspace(_build_square_level())
        workspace._active_canvas_surface_drawing_vertex_id = "a" * 32

        BlueprintWorkspace._handle_canvas_surface_vertex_chain_reset_requested(
            workspace
        )

        self.assertIsNone(workspace._active_canvas_surface_drawing_vertex_id)
        workspace._sync_canvas_surface_drawing_overlay.assert_called_once_with()


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
