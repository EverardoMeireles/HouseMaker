"""Canvas/scene landing transactions, delayed rebuilds and structural generation."""

# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
import trimesh
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from shapely.geometry import Polygon
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import Z_UP_TO_GLTF_Y_UP_TRANSFORM, GeneratedModel, convert_to_glb
from housemaker.landing_generation import LandingMeshGenerationResult
from housemaker.landing_geometry import (
    attach_stair_point_to_landing,
    build_landing_mesh,
    build_landing_meshes,
    landing_image_footprint,
    landing_top_world_z,
    resolve_stair_section_image_points,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.main import BlueprintWorkspace
from housemaker.models import LandingData, LevelData, StairData
from housemaker.project_io import load_project, save_project
from housemaker.viewer import TRANSFORM_GIZMO_TRANSLATE, _TransformGizmoHandle

# ### Module state ###
_application = QApplication.instance() or QApplication([])
_application.setQuitOnLastWindowClosed(False)


# ### Workspace fixtures ###
def _mesh_result(polygon=None, *, color=(180, 80, 30), texture_only=False):
    """Build a textured provider-shaped result without making an API request."""

    level = LevelData(index=2, name="Meshy mock")
    values = {} if polygon is None else {"footprint_meters": polygon}
    landing = LandingData(center_x=0, center_y=0, thickness_meters=0.3, **values)
    mesh = build_landing_mesh(level, landing, [level])
    mesh.apply_translation((0, 0, -mesh.bounds[0, 2]))
    mesh.visual = TextureVisuals(
        uv=np.asarray(mesh.vertices[:, :2]) / 1.5 + 0.5,
        material=PBRMaterial(baseColorTexture=Image.new("RGB", (4, 4), color)),
    )
    exported = mesh.copy()
    exported.apply_transform(Z_UP_TO_GLTF_Y_UP_TRANSFORM)
    scene = trimesh.Scene(exported)
    model = GeneratedModel(
        mesh=mesh, scene=scene, glb_bytes=scene.export(file_type="glb")
    )
    return LandingMeshGenerationResult(model, "mock-task", texture_only=texture_only)


class LandingMainIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary.name) / "settings.json"
            )
        )
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        self.workspace.generation._asset_directory = (
            Path(self.temporary.name) / "generated"
        )
        self.level = self.workspace.current_level
        self.level.image_size_pixels = (500.0, 500.0)
        self.workspace._sync_canvas_to_current_level()
        self.workspace.canvas.blueprint_image = QImage(
            500, 500, QImage.Format.Format_RGB32
        )
        self.workspace.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        self.workspace.canvas.set_stair_context(self.workspace.stairs, self.level)
        self.rebuild_patch = patch.object(
            self.workspace, "_reconcile_canvas_surface_edit_and_refresh"
        )
        self.rebuild = self.rebuild_patch.start()
        self.workspace._canvas_undo_stack.clear()

    def tearDown(self):
        self.workspace.shutdown()
        self.workspace.close()
        _application.processEvents()
        self.rebuild_patch.stop()
        self.temporary.cleanup()

    def _add(self, x=100.0, y=100.0):
        self.workspace.canvas.landing_placement_requested.emit(x, y)
        landing = self.level.landings[-1]
        self.workspace._commit_landing_mesh_update()
        self.workspace._canvas_undo_stack.clear()
        self.rebuild.reset_mock()
        return landing

    def _attach_stair(self, landing):
        level_index = self.level.index
        stair = StairData(
            start_level_index=level_index,
            end_level_index=level_index + 1,
            start_a_x=65,
            start_a_y=80,
            start_b_x=65,
            start_b_y=120,
            end_a_x=220,
            end_a_y=80,
            end_b_x=220,
            end_b_y=120,
            stringer_placement="none",
        )
        for name in ("a", "b"):
            stair = attach_stair_point_to_landing(stair, 0, name, self.level, landing)
        self.workspace.stairs = [stair]
        self.workspace.canvas.set_stair_context(self.workspace.stairs, self.level)
        return stair

    # ### Placement modes and independent scene signals ###
    def test_canvas_placement_selects_landing_and_waits_for_mesh_delay(self):
        self.workspace.canvas.landing_placement_requested.emit(120.0, 140.0)
        self.assertEqual(len(self.level.landings), 1)
        landing = self.level.landings[0]
        self.assertEqual((landing.center_x, landing.center_y), (120.0, 140.0))
        self.assertEqual(self.workspace._selected_landing_id, landing.landing_id)
        self.assertEqual(self.workspace.canvas.selected_landing_id, landing.landing_id)
        self.assertTrue(self.workspace._pending_landing_mesh_update)
        self.assertTrue(self.workspace._landing_mesh_update_timer.isActive())
        self.rebuild.assert_not_called()
        self.workspace._commit_landing_mesh_update()
        self.rebuild.assert_called_once()

    def test_scene_placement_converts_world_position_into_owning_level_pixels(self):
        world_xy = level_image_to_world_xy(self.level, 150, 160)
        self.workspace.viewer.landing_placement_requested.emit(
            {"level_index": self.level.index, "point": (*world_xy, 10.0)}
        )
        landing = self.level.landings[0]
        self.assertAlmostEqual(landing.center_x, 150.0)
        self.assertAlmostEqual(landing.center_y, 160.0)
        self.assertEqual(landing.elevation_meters, 0.0)
        self.assertTrue(self.workspace._pending_landing_mesh_update)

    def test_add_button_arms_both_views_and_cancel_stops_both_without_adding(self):
        self.workspace.add_landing_button.click()
        self.assertTrue(self.workspace.canvas.is_landing_placement_active())
        self.assertTrue(self.workspace.viewer.is_landing_placement_active())
        self.workspace.canvas.landing_placement_cancelled.emit()
        self.assertFalse(self.workspace.canvas.is_landing_placement_active())
        self.assertFalse(self.workspace.viewer.is_landing_placement_active())
        self.assertEqual(self.level.landings, [])

    # ### Live transforms, delay and shared undo ###
    def test_scene_transform_previews_update_footprints_and_attached_stairs_live(self):
        landing = self._add()
        stair = self._attach_stair(landing)
        previous_points = landing_image_footprint(self.level, landing)
        self.workspace.viewer.landing_edit_started.emit(landing.landing_id)
        self.workspace.viewer.landing_transform_preview_changed.emit(
            landing.landing_id,
            {
                "center_x": 130.0,
                "center_y": 150.0,
                "rotation_degrees": 90.0,
                "scale_x": 2.0,
                "scale_y": 0.5,
                "scale_z": 2.0,
            },
        )
        updated = self.level.landings[0]
        self.assertNotEqual(
            landing_image_footprint(self.level, updated), previous_points
        )
        self.assertEqual(
            resolve_stair_section_image_points(self.level, stair.start_section)[0],
            (140.0, 80.0),
        )
        self.assertEqual(
            self.workspace.canvas._resolve_stair_canvas_point(
                65, 80, None, stair.start_a_landing
            ),
            (140.0, 80.0),
        )
        self.assertIn(
            f"landing:{landing.landing_id}", self.workspace.viewer._landing_previews
        )
        committed = next(
            level
            for level in self.workspace._build_viewer_preview_levels()
            if level.index == self.level.index
        )
        self.assertEqual(committed.landings, [landing])
        self.assertFalse(self.workspace._landing_mesh_update_timer.isActive())
        self.rebuild.assert_not_called()
        self.workspace.viewer.landing_edit_finished.emit(landing.landing_id, True)
        self.assertTrue(self.workspace._landing_mesh_update_timer.isActive())
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.rebuild.assert_not_called()
        self.workspace._commit_landing_mesh_update()
        self.rebuild.assert_called_once()

    def test_canvas_transform_undo_restores_landing_and_stair_anchors(self):
        landing = self._add()
        stair = self._attach_stair(landing)
        self.workspace.canvas.landing_edit_started.emit(landing.landing_id)
        self.workspace.canvas.landing_transform_preview_changed.emit(
            landing.landing_id, {"center_x": 160.0, "rotation_degrees": 35.0}
        )
        self.workspace.canvas.landing_edit_finished.emit(landing.landing_id, True)
        self.workspace.canvas.undo_requested.emit()
        self.assertEqual(self.level.landings, [landing])
        self.assertEqual(self.workspace.stairs, [stair])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_cancel_transform_restores_geometry_without_recording_undo(self):
        landing = self._add()
        self.workspace.viewer.landing_edit_started.emit(landing.landing_id)
        self.workspace.viewer.landing_transform_preview_changed.emit(
            landing.landing_id, {"elevation_meters": 1.5, "scale_z": 3.0}
        )
        self.workspace.viewer.landing_edit_cancelled.emit(landing.landing_id)
        self.assertEqual(self.level.landings, [landing])
        self.assertFalse(self.workspace._pending_landing_mesh_update)
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.rebuild.assert_not_called()

    def test_invalid_transform_is_rejected_without_altering_landing(self):
        landing = self._add()
        self.workspace.canvas.landing_edit_started.emit(landing.landing_id)
        self.workspace.canvas.landing_transform_preview_changed.emit(
            landing.landing_id, {"scale_x": 0.0}
        )
        self.workspace.canvas.landing_edit_finished.emit(landing.landing_id, False)
        self.assertEqual(self.level.landings, [landing])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    # ### Rapid consecutive edits and paused mesh rebuilds ###
    def test_pending_placement_cannot_rebuild_and_clear_a_new_scene_gizmo_drag(self):
        self.workspace.canvas.landing_placement_requested.emit(120.0, 140.0)
        landing = self.level.landings[0]
        viewer = self.workspace.viewer
        viewer.set_model(convert_to_glb(self.workspace.levels))
        viewer.set_landing_context(self.workspace.levels, landing.landing_id)
        group = viewer._placed_object_render_groups[f"landing:{landing.landing_id}"]
        pivot = np.asarray(group.preview.world_position)
        offset = np.asarray((3.0, -5.0, 4.0))
        ray = pivot + offset, -offset / np.linalg.norm(offset)

        # Model replacement normally discards the active gizmo transaction.
        self.rebuild.side_effect = lambda: setattr(
            viewer, "_placed_object_transform_drag", None
        )
        with patch.object(viewer.view, "build_camera_ray", return_value=ray):
            started = viewer._begin_placed_object_gizmo_drag(
                _TransformGizmoHandle(TRANSFORM_GIZMO_TRANSLATE, 0), QPointF()
            )

        self.assertTrue(started)
        self.assertIsNotNone(viewer._placed_object_transform_drag)
        moved_ray = ray[0] + np.asarray((0.25, 0.0, 0.0)), ray[1]
        with patch.object(viewer.view, "build_camera_ray", return_value=moved_ray):
            self.assertTrue(viewer._update_placed_object_gizmo_drag(QPointF()))
        self.assertNotEqual(self.level.landings[0].center_x, landing.center_x)
        self.assertFalse(self.workspace._landing_mesh_update_timer.isActive())
        self.rebuild.assert_not_called()
        viewer._cancel_placed_object_gizmo_drag()
        self.assertTrue(self.workspace._landing_mesh_update_timer.isActive())

    def test_delayed_commit_is_paused_until_the_active_landing_drag_finishes(self):
        landing = self._add()
        self.workspace.viewer.landing_edit_started.emit(landing.landing_id)
        self.assertTrue(self.workspace._pending_landing_mesh_update)
        self.workspace.viewer.landing_transform_preview_changed.emit(
            landing.landing_id, {"center_x": 130.0}
        )

        self.workspace._commit_landing_mesh_update()

        self.assertTrue(self.workspace._pending_landing_mesh_update)
        self.assertFalse(self.workspace._landing_mesh_update_timer.isActive())
        self.assertEqual(
            self.workspace._viewer_landings_by_level_index[self.level.index],
            (landing,),
        )
        self.rebuild.assert_not_called()
        self.workspace.viewer.landing_edit_finished.emit(landing.landing_id, True)
        self.assertTrue(self.workspace._landing_mesh_update_timer.isActive())
        self.workspace._commit_landing_mesh_update()
        self.rebuild.assert_called_once()
        self.assertEqual(
            self.workspace._viewer_landings_by_level_index[self.level.index],
            tuple(self.level.landings),
        )

    def test_cancel_of_second_drag_resumes_the_previous_uncommitted_transform(self):
        landing = self._add()
        viewer = self.workspace.viewer
        viewer.landing_edit_started.emit(landing.landing_id)
        viewer.landing_transform_preview_changed.emit(
            landing.landing_id, {"center_x": 130.0}
        )
        viewer.landing_edit_finished.emit(landing.landing_id, True)
        previous_edit = self.level.landings[0]
        viewer.landing_edit_started.emit(landing.landing_id)
        viewer.landing_transform_preview_changed.emit(
            landing.landing_id, {"center_x": 170.0, "scale_z": 1.5}
        )

        viewer.landing_edit_cancelled.emit(landing.landing_id)

        self.assertEqual(self.level.landings, [previous_edit])
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.assertTrue(self.workspace._pending_landing_mesh_update)
        self.assertTrue(self.workspace._landing_mesh_update_timer.isActive())
        self.assertEqual(
            self.workspace._viewer_landings_by_level_index[self.level.index],
            (landing,),
        )
        self.rebuild.assert_not_called()
        self.workspace._commit_landing_mesh_update()
        self.rebuild.assert_called_once()
        self.assertEqual(
            self.workspace._viewer_landings_by_level_index[self.level.index],
            (previous_edit,),
        )

    def test_no_op_second_drag_resumes_the_previous_uncommitted_transform(self):
        landing = self._add()
        viewer = self.workspace.viewer
        viewer.landing_edit_started.emit(landing.landing_id)
        viewer.landing_transform_preview_changed.emit(
            landing.landing_id, {"center_x": 130.0}
        )
        viewer.landing_edit_finished.emit(landing.landing_id, True)
        previous_edit = self.level.landings[0]

        viewer.landing_edit_started.emit(landing.landing_id)
        viewer.landing_edit_finished.emit(landing.landing_id, False)

        self.assertEqual(self.level.landings, [previous_edit])
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.assertTrue(self.workspace._pending_landing_mesh_update)
        self.assertTrue(self.workspace._landing_mesh_update_timer.isActive())
        self.rebuild.assert_not_called()
        self.workspace._commit_landing_mesh_update()
        self.rebuild.assert_called_once()

    def test_no_op_drag_without_pending_work_does_not_schedule_a_mesh_rebuild(self):
        landing = self._add()
        viewer = self.workspace.viewer
        viewer.landing_edit_started.emit(landing.landing_id)
        viewer.landing_edit_finished.emit(landing.landing_id, False)

        self.assertFalse(self.workspace._pending_landing_mesh_update)
        self.assertFalse(self.workspace._landing_mesh_update_timer.isActive())
        self.assertEqual(self.workspace._canvas_undo_stack, [])
        self.rebuild.assert_not_called()

    def test_mesh_update_queued_during_a_drag_does_not_start_its_timer(self):
        landing = self._add()
        self.workspace.canvas.landing_edit_started.emit(landing.landing_id)

        self.workspace._queue_landing_mesh_update()

        self.assertTrue(self.workspace._pending_landing_mesh_update)
        self.assertFalse(self.workspace._landing_mesh_update_timer.isActive())
        self.workspace._commit_landing_mesh_update()
        self.rebuild.assert_not_called()
        self.workspace.canvas.landing_edit_cancelled.emit(landing.landing_id)

    # ### Structural generation and persistence ###
    def test_generated_concave_shape_preserves_transforms_and_has_single_undo(self):
        original = self._add()
        landing = replace(
            original,
            center_x=130,
            center_y=140,
            rotation_degrees=25,
            scale_x=1.4,
            scale_y=0.8,
            elevation_meters=1.1,
        )
        self.level.landings[0] = landing
        stair = self._attach_stair(original)
        polygon = (
            (-0.75, -0.75),
            (0.75, -0.75),
            (0.75, 0),
            (0, 0),
            (0, 0.75),
            (-0.75, 0.75),
        )
        self.workspace.generation.landing_generation_completed.emit(
            landing.landing_id, _mesh_result(polygon)
        )
        updated = self.level.landings[0]
        self.assertTrue(
            Polygon(updated.footprint_meters).equals_exact(
                Polygon(polygon), 1e-5, normalize=True
            )
        )
        self.assertAlmostEqual(updated.thickness_meters, 0.3)
        self.assertEqual(updated.center_x, landing.center_x)
        self.assertEqual(updated.center_y, landing.center_y)
        self.assertEqual(updated.rotation_degrees, landing.rotation_degrees)
        self.assertEqual(updated.scale_x, landing.scale_x)
        self.assertEqual(updated.scale_y, landing.scale_y)
        self.assertEqual(updated.elevation_meters, landing.elevation_meters)
        self.assertTrue(Path(updated.generated_model_path).is_file())
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)
        self.assertTrue(self.workspace._pending_landing_mesh_update)
        self.rebuild.assert_not_called()
        self.workspace.canvas.undo_requested.emit()
        self.assertEqual(self.level.landings, [landing])
        self.assertEqual(self.workspace.stairs, [stair])

    def test_invalid_generated_mesh_and_stale_identity_are_ignored(self):
        landing = self._add()
        invalid = _mesh_result()
        invalid.model.glb_bytes = b"invalid glb"
        self.workspace.generation.landing_generation_completed.emit(
            landing.landing_id, invalid
        )
        self.assertEqual(self.level.landings, [landing])
        self.assertIn("rejected", self.workspace.stair_status_label.text())
        self.workspace.generation.landing_generation_completed.emit(
            "f" * 32, _mesh_result()
        )
        self.assertEqual(self.level.landings, [landing])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_meshy_landing_textures_survive_preview_save_reload_and_export(self):
        landing = self._add()
        self.workspace.viewer.set_model(convert_to_glb(self.workspace.levels))
        self.workspace._sync_landing_views()
        self.workspace.generation.landing_generation_completed.emit(
            landing.landing_id, _mesh_result()
        )
        updated = self.level.landings[0]
        self.workspace.canvas.landing_generation_requested.emit(landing.landing_id)
        preview = self.workspace.generation.result_view.model
        self.assertTrue(preview.glb_bytes)
        self.assertEqual(len(preview.mesh.faces), 12)
        self.assertTrue(preview.scene.metadata["housemaker_landing_preview"])
        self.assertTrue(self.workspace.generation.result_view.model_material_items)
        group = self.workspace.viewer._placed_object_render_groups[
            f"landing:{landing.landing_id}"
        ]
        self.assertEqual(group.preview.meshes[0].visual.kind, "texture")
        self.assertTrue(
            any(part.textured_item is not None for part in group.retained_parts)
        )
        path = Path(self.temporary.name) / "mesh-project.json"
        save_project(path, self.workspace.current_level_index, self.workspace.levels)
        restored = load_project(path)
        level = next(item for item in restored.levels if item.index == self.level.index)
        self.assertEqual(level.landings[0], updated)
        meshes = build_landing_meshes(level, level.landings[0], restored.levels)
        self.assertEqual(meshes[0].visual.kind, "texture")
        exported = convert_to_glb(restored.levels)
        self.assertTrue(
            any(
                mesh.visual.kind == "texture"
                for mesh in exported.scene.geometry.values()
            )
        )

    def test_texture_only_landing_generation_keeps_profile_and_stair_anchors(self):
        landing = self._add()
        self.workspace.generation.landing_generation_completed.emit(
            landing.landing_id, _mesh_result()
        )
        original = self.level.landings[0]
        stair = self._attach_stair(original)
        self.workspace.generation.landing_generation_completed.emit(
            landing.landing_id, _mesh_result(color=(40, 100, 220), texture_only=True)
        )
        updated = self.level.landings[0]
        self.assertNotEqual(updated.generated_model_path, original.generated_model_path)
        self.assertEqual(
            replace(updated, generated_model_path=original.generated_model_path),
            original,
        )
        self.assertEqual(self.workspace.stairs, [stair])

    def test_generation_control_switches_to_dedicated_landing_mode(self):
        landing = self._add()
        self.workspace.canvas.landing_generation_requested.emit(landing.landing_id)
        self.assertEqual(
            self.workspace.generation.get_landing_editing_target(), landing
        )
        self.assertEqual(
            self.workspace.workspace_tabs.currentIndex(),
            self.workspace.generation_workspace_tab_index,
        )

    def test_connected_landing_cannot_be_deleted_leaving_dangling_stairs(self):
        landing = self._add()
        stair = self._attach_stair(landing)
        self.workspace.viewer.landing_deletion_requested.emit(landing.landing_id)
        self.assertEqual(self.level.landings, [landing])
        self.assertEqual(self.workspace.stairs, [stair])
        self.assertIn("Disconnect", self.workspace.stair_status_label.text())

    def test_workspace_saved_landing_and_stair_links_export_after_roundtrip(self):
        landing = self._add()
        stair = self._attach_stair(landing)
        path = Path(self.temporary.name) / "project.json"
        save_project(
            path,
            self.workspace.current_level_index,
            self.workspace.levels,
            stairs=self.workspace.stairs,
        )
        restored = load_project(path)
        level = next(item for item in restored.levels if item.index == self.level.index)
        self.assertEqual(level.landings, [landing])
        self.assertEqual(restored.stairs, [stair])
        model = convert_to_glb(restored.levels, stairs=restored.stairs)
        self.assertIn(f"landing:{landing.landing_id}", model.scene.geometry)
        self.assertGreater(landing_top_world_z(level, landing, restored.levels), 0.0)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
