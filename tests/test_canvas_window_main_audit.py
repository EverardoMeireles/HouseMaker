# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.glb import convert_to_glb
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.surface_geometry import (
    add_wall_window,
    build_fixed_surfaces,
    build_wall_window_placement,
)

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
    center = vertex_data.add_vertex(50.0, 50.0)
    room = RoomData(
        name="Living room",
        vertex_ids=boundary_ids,
        center_vertex_id=center.id,
        color_rgb=(140, 180, 220),
    )
    return LevelData(
        index=2,
        name="Ground",
        vertex_data=vertex_data,
        rooms=[room],
    )


def _get_target_wall(level: LevelData):
    return next(
        surface
        for surface in build_fixed_surfaces([level])
        if surface.surface_id == "level:2/room:5/wall:1:2"
    )


def _add_outer_wall(level: LevelData) -> str:
    start = level.vertex_data.add_vertex(0.0, -10.0)
    end = level.vertex_data.add_vertex(100.0, -10.0)
    level.vertex_data.add_edge(start.id, end.id)
    return (
        f"level:{level.index}/wall:{min(start.id, end.id)}:"
        f"{max(start.id, end.id)}"
    )


def _mesh_covers_point_on_plane(
    mesh: object,
    point: tuple[float, float, float],
    fixed_axis: int,
) -> bool:
    point_array = np.asarray(point, dtype=float)
    plane_axes = [axis for axis in range(3) if axis != fixed_axis]
    for triangle in np.asarray(getattr(mesh, "triangles"), dtype=float):
        if not np.allclose(
            triangle[:, fixed_axis],
            point_array[fixed_axis],
            atol=1e-6,
        ):
            continue
        if _point_is_in_triangle_2d(
            point_array[plane_axes],
            triangle[:, plane_axes],
        ):
            return True
    return False


def _point_is_in_triangle_2d(
    point: np.ndarray,
    triangle: np.ndarray,
) -> bool:
    first, second, third = triangle
    denominator = (
        (second[1] - third[1]) * (first[0] - third[0])
        + (third[0] - second[0]) * (first[1] - third[1])
    )
    if abs(float(denominator)) <= 1e-9:
        return False
    first_weight = (
        (second[1] - third[1]) * (point[0] - third[0])
        + (third[0] - second[0]) * (point[1] - third[1])
    ) / denominator
    second_weight = (
        (third[1] - first[1]) * (point[0] - third[0])
        + (first[0] - third[0]) * (point[1] - third[1])
    ) / denominator
    third_weight = 1.0 - first_weight - second_weight
    return min(first_weight, second_weight, third_weight) >= -1e-6


def _primary_screen():
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("A QScreen is required for detached viewer tests.")
    return screen


# ### Main window workflow integration tests ###
class CanvasWindowMainAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_window_placement_waits_for_mesh_edit_delay_before_3d_rebuild(
        self,
    ) -> None:
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        level = _build_square_level()
        self.workspace.levels = [level]
        self.workspace._reset_viewer_doorway_snapshots()
        wall = _get_target_wall(level)
        placement = build_wall_window_placement(
            wall,
            (0.5, 0.0, 0.75),
            (1.5, 0.0, 2.25),
        )

        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
            ) as build_preview,
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
        ):
            self.workspace._handle_canvas_window_placement_requested(placement)

            self.assertEqual(len(level.windows), 1)
            self.assertAlmostEqual(
                _get_target_wall(level).area_square_meters,
                4.5,
            )
            preview_level = self.workspace._build_viewer_preview_levels()[0]
            self.assertEqual(preview_level.windows, [])
            self.assertAlmostEqual(
                _get_target_wall(preview_level).area_square_meters,
                6.0,
            )
            self.assertEqual(
                self.workspace._pending_window_mesh_level_index,
                level.index,
            )
            self.assertTrue(
                self.workspace._doorway_mesh_update_timer.isActive()
            )
            build_preview.assert_not_called()
            schedule_refresh.assert_not_called()

            self.workspace._commit_pending_doorway_mesh_update()

        self.assertFalse(self.workspace._doorway_mesh_update_timer.isActive())
        self.assertIsNone(self.workspace._pending_window_mesh_level_index)
        self.assertEqual(
            len(self.workspace._build_viewer_preview_levels()[0].windows),
            1,
        )
        build_preview.assert_not_called()
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertEqual(
            self.workspace._canvas_window_undo_ids,
            [level.windows[0].window_id],
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), 1)

    def test_window_undo_is_lifo_across_successful_commits(self) -> None:
        level = _build_square_level()
        self.workspace.levels = [level]
        first_wall = _get_target_wall(level)
        first_placement = build_wall_window_placement(
            first_wall,
            (0.15, 0.0, 0.3),
            (0.55, 0.0, 0.8),
        )
        self.workspace._handle_canvas_window_placement_requested(first_placement)
        first_window = level.windows[0]
        second_wall = _get_target_wall(level)
        second_placement = build_wall_window_placement(
            second_wall,
            (1.0, 0.0, 1.0),
            (1.5, 0.0, 2.0),
        )
        self.workspace._handle_canvas_window_placement_requested(second_placement)
        second_window = level.windows[1]

        self.workspace.viewer.undo_requested.emit()

        self.assertEqual(level.windows, [first_window])
        self.assertNotIn(second_window, level.windows)
        self.assertEqual(
            self.workspace._canvas_window_undo_ids,
            [first_window.window_id],
        )

        self.workspace.viewer.undo_requested.emit()

        self.assertEqual(level.windows, [])
        self.assertEqual(self.workspace._canvas_window_undo_ids, [])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_window_undo_restores_both_wall_faces_and_removes_reveal(
        self,
    ) -> None:
        level = _build_square_level()
        outer_surface_id = _add_outer_wall(level)
        self.workspace.levels = [level]
        wall = _get_target_wall(level)
        placement = build_wall_window_placement(
            wall,
            (0.5, 0.0, 0.75),
            (1.5, 0.0, 2.25),
        )
        inner_center = (1.0, 0.0, 1.5)
        outer_center = (1.0, 0.2, 1.5)

        self.workspace._handle_canvas_window_placement_requested(placement)
        self.workspace._commit_pending_doorway_mesh_update()

        cut_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
        }
        model = self.workspace._build_viewer_preview_model(None)
        self.assertIsNotNone(model)
        assert model is not None
        self.assertTrue(
            any(
                name.endswith("_window_reveals")
                for name in model.scene.geometry
            )
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                cut_surfaces[wall.surface_id].mesh,
                inner_center,
                fixed_axis=1,
            )
        )
        self.assertFalse(
            _mesh_covers_point_on_plane(
                cut_surfaces[outer_surface_id].mesh,
                outer_center,
                fixed_axis=1,
            )
        )

        self.workspace.viewer.undo_requested.emit()

        restored_surfaces = {
            surface.surface_id: surface
            for surface in build_fixed_surfaces([level])
        }
        restored_model = self.workspace._build_viewer_preview_model(None)
        self.assertIsNotNone(restored_model)
        assert restored_model is not None
        self.assertTrue(
            _mesh_covers_point_on_plane(
                restored_surfaces[wall.surface_id].mesh,
                inner_center,
                fixed_axis=1,
            )
        )
        self.assertTrue(
            _mesh_covers_point_on_plane(
                restored_surfaces[outer_surface_id].mesh,
                outer_center,
                fixed_axis=1,
            )
        )
        self.assertFalse(
            any(
                name.endswith("_window_reveals")
                for name in restored_model.scene.geometry
            )
        )

    def test_failed_window_undo_restores_exact_index_and_history(self) -> None:
        for failure_kind in ("none", "exception"):
            with self.subTest(failure_kind=failure_kind):
                self.workspace._canvas_window_undo_ids.clear()
                self.workspace.viewer.set_window_undo_available(False)
                level = _build_square_level()
                existing_wall = _get_target_wall(level)
                existing_placement = build_wall_window_placement(
                    existing_wall,
                    (0.15, 0.0, 0.3),
                    (0.55, 0.0, 0.8),
                )
                existing_window = add_wall_window(
                    [level],
                    existing_placement,
                    window_id=f"existing-{failure_kind}",
                )
                self.workspace.levels = [level]
                added_wall = _get_target_wall(level)
                added_placement = build_wall_window_placement(
                    added_wall,
                    (1.0, 0.0, 1.0),
                    (1.5, 0.0, 2.0),
                )
                self.workspace._handle_canvas_window_placement_requested(
                    added_placement
                )
                added_window = level.windows[1]
                history_before = list(self.workspace._canvas_window_undo_ids)
                windows_before = list(level.windows)
                build_patch = (
                    patch.object(
                        self.workspace,
                        "_build_viewer_preview_model",
                        return_value=None,
                    )
                    if failure_kind == "none"
                    else patch.object(
                        self.workspace,
                        "_build_viewer_preview_model",
                        side_effect=RuntimeError("undo refresh failure"),
                    )
                )
                with (
                    build_patch,
                    patch.object(
                        self.workspace.viewer,
                        "set_model",
                    ) as set_model,
                    patch.object(
                        self.workspace.viewer,
                        "set_wall_targets",
                    ) as set_wall_targets,
                    patch.object(
                        self.workspace.surface_texture_generation,
                        "set_levels",
                    ) as set_surface_levels,
                    patch.object(
                        self.workspace.surface_texture_generation,
                        "set_preview_model",
                    ) as set_preview_model,
                ):
                    self.workspace._handle_canvas_window_undo_requested()

                self.assertEqual(level.windows, windows_before)
                self.assertIs(level.windows[0], existing_window)
                self.assertIs(level.windows[1], added_window)
                self.assertEqual(
                    self.workspace._canvas_window_undo_ids,
                    history_before,
                )
                set_model.assert_not_called()
                set_wall_targets.assert_not_called()
                set_surface_levels.assert_not_called()
                set_preview_model.assert_not_called()

    def test_window_placement_does_not_build_model_before_delay(self) -> None:
        level = _build_square_level()
        first_wall = _get_target_wall(level)
        first_placement = build_wall_window_placement(
            first_wall,
            (0.15, 0.0, 0.3),
            (0.55, 0.0, 0.8),
        )
        existing_window = add_wall_window(
            [level],
            first_placement,
            window_id="existing-window",
        )
        second_wall = _get_target_wall(level)
        second_placement = build_wall_window_placement(
            second_wall,
            (1.0, 0.0, 1.0),
            (1.5, 0.0, 2.0),
        )
        self.workspace.levels = [level]

        with (
            patch.object(
                self.workspace,
                "_build_viewer_preview_model",
                side_effect=RuntimeError("must not build before delay"),
            ) as build_preview,
            patch.object(self.workspace.viewer, "set_model") as set_model,
            patch.object(
                self.workspace.viewer,
                "set_wall_targets",
            ) as set_wall_targets,
            patch.object(
                self.workspace.surface_texture_generation,
                "set_preview_model",
            ) as set_preview_model,
        ):
            self.workspace._handle_canvas_window_placement_requested(
                second_placement
            )

        self.assertEqual(len(level.windows), 2)
        self.assertIs(level.windows[0], existing_window)
        self.assertEqual(
            self.workspace._pending_window_mesh_level_index,
            level.index,
        )
        self.assertTrue(self.workspace._doorway_mesh_update_timer.isActive())
        build_preview.assert_not_called()
        set_model.assert_not_called()
        set_wall_targets.assert_not_called()
        set_preview_model.assert_not_called()

    def test_project_replacement_keeps_loaded_windows_but_clears_undo_history(
        self,
    ) -> None:
        session_level = _build_square_level()
        self.workspace.levels = [session_level]
        session_wall = _get_target_wall(session_level)
        self.workspace._handle_canvas_window_placement_requested(
            build_wall_window_placement(
                session_wall,
                (0.5, 0.0, 0.75),
                (1.5, 0.0, 2.25),
            )
        )
        self.assertTrue(self.workspace._canvas_window_undo_ids)

        loaded_level = _build_square_level()
        loaded_wall = _get_target_wall(loaded_level)
        loaded_window = add_wall_window(
            [loaded_level],
            build_wall_window_placement(
                loaded_wall,
                (0.5, 0.0, 0.75),
                (1.5, 0.0, 2.25),
            ),
            window_id="loaded-window",
        )
        self.workspace._apply_project_state(
            levels=[loaded_level],
            current_level_index=0,
        )

        self.assertEqual(loaded_level.windows, [loaded_window])
        self.assertEqual(self.workspace._canvas_window_undo_ids, [])
        self.assertEqual(self.workspace._canvas_undo_stack, [])

    def test_canvas_window_button_arms_both_available_views(self) -> None:
        level = _build_square_level()
        self.workspace.levels = [level]
        self.workspace.current_level_index = 0
        self.workspace._sync_canvas_to_current_level()
        self.workspace.canvas.blueprint_image = QImage(
            100,
            100,
            QImage.Format.Format_RGB32,
        )
        self.workspace.viewer.set_wall_targets(
            tuple(build_fixed_surfaces([level]))
        )
        scene_button = self.workspace.viewer.add_window_frame_button
        assert scene_button is not None

        self.workspace.add_window_button.click()

        self.assertTrue(self.workspace.add_window_button.isChecked())
        self.assertTrue(scene_button.isChecked())
        self.assertTrue(self.workspace.canvas.is_window_placement_active())
        self.assertTrue(self.workspace.viewer.is_window_placement_active())

        self.workspace.viewer.cancel_window_placement()

        self.assertFalse(self.workspace.add_window_button.isChecked())
        self.assertFalse(scene_button.isChecked())
        self.assertFalse(self.workspace.canvas.is_window_placement_active())
        self.assertFalse(self.workspace.viewer.is_window_placement_active())

    def test_detached_scene_contains_its_window_frame_button(
        self,
    ) -> None:
        level = _build_square_level()
        self.workspace.levels = [level]
        model = convert_to_glb([level])
        wall = _get_target_wall(level)
        viewer = self.workspace.viewer
        viewer.set_model(model)
        self.workspace._remember_current_canvas_preview_model(
            model,
            validated_dependency_signature=(
                self.workspace._build_viewer_preview_dependency_signature()
            ),
        )
        viewer.set_wall_targets(tuple(build_fixed_surfaces([level])))
        self.assertTrue(viewer.select_wall_target(wall.surface_id))
        panel = viewer.window_tools_panel
        button = self.workspace.add_window_button
        scene_button = viewer.add_window_frame_button
        undo_button = viewer.undo_window_button
        self.assertIs(panel, viewer.canvas_tools_panel)
        assert panel is not None
        self.assertIsNotNone(scene_button)
        assert scene_button is not None
        self.assertIsNone(undo_button)
        self.assertIsNone(viewer.window_tools_status_label)
        self.assertIsNone(
            panel.findChild(type(button), "canvas-add-window-button")
        )
        self.assertIs(
            panel.findChild(
                type(scene_button),
                "canvas-scene-add-window-frame-button",
            ),
            scene_button,
        )
        self.assertFalse(viewer.isAncestorOf(button))
        self.assertTrue(viewer.isAncestorOf(scene_button))

        screen = _primary_screen()
        with patch(
            "housemaker.main.resolve_fullscreen_3d_viewer_screen",
            return_value=screen,
        ):
            self.workspace._apply_scene_3d_display_screen("audit-screen")
        _qt_application.processEvents()

        host = self.workspace._external_scene_3d_host
        self.assertTrue(host.is_active)
        self.assertIs(host.viewer, self.workspace.scene_3d_workspace)
        self.assertIs(
            self.workspace.scene_3d_workspace.parentWidget(),
            host.window,
        )
        self.assertIs(viewer.parentWidget(), self.workspace.scene_3d_workspace)
        self.assertIs(viewer.window_tools_panel, panel)
        self.assertTrue(viewer.isAncestorOf(panel))
        self.assertTrue(panel.isVisibleTo(host.window))
        self.assertGreaterEqual(panel.geometry().left(), viewer.view.width())
        self.assertIsNone(viewer.undo_window_button)
        self.assertEqual(button.text(), "Add window frame")
        self.assertEqual(scene_button.text(), "Add window frame")
        self.assertTrue(button.isEnabled())
        self.assertTrue(scene_button.isVisibleTo(host.window))
        self.assertTrue(scene_button.isEnabled())

        self.workspace._apply_scene_3d_display_screen(None)
        _qt_application.processEvents()

        self.assertFalse(host.is_active)
        self.assertIs(
            self.workspace.workspace_tabs.widget(
                self.workspace.scene_3d_workspace_tab_index
            ),
            self.workspace.scene_3d_workspace,
        )
        self.assertIs(viewer.window_tools_panel, panel)
        self.assertIsNone(viewer.undo_window_button)
        self.assertFalse(viewer.isAncestorOf(button))
        self.assertTrue(viewer.isAncestorOf(scene_button))
        self.assertEqual(viewer.get_selected_wall_surface_id(), wall.surface_id)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
