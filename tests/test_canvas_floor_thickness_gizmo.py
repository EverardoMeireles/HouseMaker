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
from OpenGL import GL
from PySide6.QtWidgets import QApplication
import trimesh

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
    CanvasSurfaceEdit,
    CanvasSurfaceEditHandleTarget,
    CanvasSurfaceEditReference,
    build_canvas_surface_edit_targets,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, VertexData
from housemaker.surface_geometry import (
    SURFACE_TYPE_FLOOR,
    FixedSurface,
    build_fixed_surfaces,
)
from housemaker.viewer import (
    CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
    GlbViewerWidget,
    _CanvasSurfaceEditDrag,
    _build_canvas_surface_edit_outline_positions,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_square_level(
    level_index: int = 2,
    *,
    floor_thickness_meters: float = 0.3,
) -> LevelData:
    vertex_data = VertexData()
    vertices = tuple(
        vertex_data.add_vertex(*point)
        for point in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
        )
    )
    for first, second in zip(vertices, (*vertices[1:], vertices[0])):
        vertex_data.add_edge(first.id, second.id)
    return LevelData(
        index=level_index,
        name=f"Level {level_index}",
        height_meters=3.0,
        floor_thickness_meters=floor_thickness_meters,
        vertex_data=vertex_data,
    )


def _build_floor_surface() -> FixedSurface:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            (
                (0.0, 0.0, 0.3),
                (4.0, 0.0, 0.3),
                (4.0, 3.0, 0.3),
                (0.0, 3.0, 0.3),
            ),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id="level:2/floor",
        surface_type=SURFACE_TYPE_FLOOR,
        level_index=2,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
    )


def _build_floor_target(
    floor: FixedSurface,
) -> CanvasSurfaceEditHandleTarget:
    return CanvasSurfaceEditHandleTarget(
        reference=CanvasSurfaceEditReference(
            kind=CANVAS_SURFACE_EDIT_FLOOR_THICKNESS,
            level_index=2,
            axis_index=2,
        ),
        surface_id=floor.surface_id,
        origin_world=(2.0, 1.5, 0.3),
        axis_world=(0.0, 0.0, 1.0),
        minimum_delta_meters=-0.29,
        maximum_delta_meters=9.7,
        baseline_value_meters=0.3,
    )


def _edit(
    target: CanvasSurfaceEditHandleTarget,
    delta_meters: float,
) -> CanvasSurfaceEdit:
    return CanvasSurfaceEdit(
        reference=target.reference,
        surface_id=target.surface_id,
        delta_meters=delta_meters,
    )


# ### Target and overlay tests ###
class CanvasFloorThicknessGizmoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def test_floor_target_raises_the_slab_top_from_its_fixed_base(self) -> None:
        level = _build_square_level()
        surfaces = tuple(build_fixed_surfaces((level,)))
        floor = next(
            surface
            for surface in surfaces
            if surface.surface_id == "level:2/floor"
        )
        targets = build_canvas_surface_edit_targets((level,), surfaces)
        target = next(
            candidate
            for candidate in targets
            if candidate.surface_id == floor.surface_id
            and candidate.reference.kind
            == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS
        )

        self.assertEqual(target.reference.key, "level:2/floor_thickness")
        self.assertEqual(target.axis_world, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(target.baseline_value_meters, 0.3)
        self.assertAlmostEqual(target.origin_world[2], 0.3)

    def test_selected_floor_shows_depth_free_upward_gizmo(self) -> None:
        floor = _build_floor_surface()
        target = _build_floor_target(floor)
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((floor,))
        viewer.set_canvas_surface_edit_targets((target,))
        self.widgets.append(viewer)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            self.assertTrue(
                viewer.select_canvas_surface_target(floor.surface_id)
            )
            viewer._refresh_canvas_surface_edit_gizmo_items()

        self.assertEqual(
            viewer.get_active_canvas_surface_id(),
            floor.surface_id,
        )
        self.assertEqual(
            viewer._get_active_canvas_surface_edit_targets(),
            (target,),
        )
        items = tuple(viewer._canvas_surface_edit_gizmo_items)
        self.assertEqual(len(items), 2)
        for item in items:
            self.assertEqual(
                item.depthValue(),
                CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
            )
            gl_options = getattr(item, "_GLGraphicsItem__glOpts")
            self.assertFalse(gl_options[GL.GL_DEPTH_TEST])
        axis_positions = np.asarray(items[0].pos, dtype=float)
        np.testing.assert_allclose(axis_positions[0], target.origin_world)
        self.assertGreater(axis_positions[1, 2], axis_positions[0, 2])

    def test_floor_drag_outline_previews_the_new_top_only(self) -> None:
        floor = _build_floor_surface()
        target = _build_floor_target(floor)
        drag = _CanvasSurfaceEditDrag(
            target=target,
            axis=np.asarray(target.axis_world, dtype=float),
            drag_plane_normal=np.asarray((0.0, 1.0, 0.0), dtype=float),
            start_axis_parameter=0.0,
            preview_delta_meters=0.2,
        )

        outline = _build_canvas_surface_edit_outline_positions(floor, drag)

        self.assertIsNotNone(outline)
        assert outline is not None
        np.testing.assert_allclose(outline[:, 2], 0.5, atol=1e-9)


# ### Main debounce integration tests ###
class CanvasFloorThicknessMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self._temporary_directory.name) / "settings.json"
            )
        )
        self.workspace.resize(1400, 850)
        self.workspace.show()
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        self.workspace._set_mesh_edit_update_delay_seconds(60.0)
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _install_level(
        self,
    ) -> tuple[LevelData, FixedSurface, CanvasSurfaceEditHandleTarget]:
        level = _build_square_level()
        self.workspace.viewer.cancel_canvas_surface_edit()
        self.workspace.levels = [level]
        self.workspace.current_level_index = 0
        self.workspace._reset_viewer_doorway_snapshots()
        self.workspace._sync_level_controls()
        self.workspace._sync_canvas_to_current_level()
        surfaces = tuple(build_fixed_surfaces((level,)))
        self.workspace._set_canvas_viewer_targets(surfaces)
        self.workspace._canvas_viewer_preview_revision = (
            self.workspace._viewer_preview_revision
        )
        floor = next(
            surface
            for surface in surfaces
            if surface.surface_id == "level:2/floor"
        )
        self.assertTrue(
            self.workspace.viewer.select_canvas_surface_target(
                floor.surface_id
            )
        )
        target = next(
            candidate
            for candidate in self.workspace._canvas_surface_edit_targets_by_key.values()
            if candidate.surface_id == floor.surface_id
            and candidate.reference.kind
            == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS
        )
        return level, floor, target

    def test_floor_drag_rebuilds_only_after_release_and_delay(self) -> None:
        level, floor, target = self._install_level()
        final_edit = _edit(target, 0.25)
        revision_before = self.workspace._viewer_preview_revision
        self.assertAlmostEqual(
            self.workspace._build_viewer_preview_levels()[
                0
            ].floor_thickness_meters,
            0.3,
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
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
        ):
            self.workspace._handle_canvas_surface_edit_started(
                _edit(target, 0.0)
            )
            self.workspace._handle_canvas_surface_edit_preview_changed(
                final_edit
            )

            self.assertFalse(
                self.workspace._canvas_surface_mesh_update_timer.isActive()
            )
            self.assertFalse(
                self.workspace._pending_canvas_surface_mesh_update
            )
            self.assertEqual(
                self.workspace._viewer_preview_revision,
                revision_before,
            )
            build_preview.assert_not_called()
            schedule_refresh.assert_not_called()
            reconcile_assignments.assert_not_called()

            self.workspace._handle_canvas_surface_edit_finished(
                final_edit,
                True,
            )

            self.assertTrue(
                self.workspace._canvas_surface_mesh_update_timer.isActive()
            )
            self.assertTrue(
                self.workspace._pending_canvas_surface_mesh_update
            )
            self.assertAlmostEqual(level.floor_thickness_meters, 0.55)
            self.assertAlmostEqual(
                self.workspace.floor_thickness_spinbox.value(),
                0.55,
            )
            self.assertAlmostEqual(
                self.workspace._build_viewer_preview_levels()[
                    0
                ].floor_thickness_meters,
                0.3,
            )
            self.assertEqual(
                self.workspace._viewer_preview_revision,
                revision_before,
            )
            build_preview.assert_not_called()
            schedule_refresh.assert_not_called()
            reconcile_assignments.assert_not_called()

            active_targets = (
                self.workspace.viewer._get_active_canvas_surface_edit_targets()
            )
            self.assertEqual(len(active_targets), 1)
            self.assertEqual(active_targets[0].surface_id, floor.surface_id)
            self.assertAlmostEqual(
                active_targets[0].baseline_value_meters,
                0.55,
            )

            self.workspace._commit_pending_canvas_surface_mesh_update()

        reconcile_assignments.assert_called_once_with([level])
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        build_preview.assert_not_called()
        self.assertAlmostEqual(
            self.workspace._build_viewer_preview_levels()[
                0
            ].floor_thickness_meters,
            0.55,
        )
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        self.assertFalse(self.workspace._pending_canvas_surface_mesh_update)

    def test_repeated_floor_drags_accumulate_before_one_mesh_commit(
        self,
    ) -> None:
        level, floor, first_target = self._install_level()

        with (
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
        ):
            first_edit = _edit(first_target, 0.25)
            self.workspace._handle_canvas_surface_edit_started(
                _edit(first_target, 0.0)
            )
            self.workspace._handle_canvas_surface_edit_preview_changed(
                first_edit
            )
            self.workspace._handle_canvas_surface_edit_finished(
                first_edit,
                True,
            )

            second_target = next(
                candidate
                for candidate in (
                    self.workspace._canvas_surface_edit_targets_by_key.values()
                )
                if candidate.surface_id == floor.surface_id
                and candidate.reference.kind
                == CANVAS_SURFACE_EDIT_FLOOR_THICKNESS
            )
            self.assertIsNot(second_target, first_target)
            self.assertAlmostEqual(second_target.baseline_value_meters, 0.55)
            self.workspace._handle_canvas_surface_edit_started(
                _edit(second_target, 0.0)
            )
            self.assertFalse(
                self.workspace._canvas_surface_mesh_update_timer.isActive()
            )

            second_edit = _edit(second_target, 0.15)
            self.workspace._handle_canvas_surface_edit_preview_changed(
                second_edit
            )
            self.workspace._handle_canvas_surface_edit_finished(
                second_edit,
                True,
            )

            self.assertAlmostEqual(level.floor_thickness_meters, 0.7)
            self.assertAlmostEqual(
                self.workspace._build_viewer_preview_levels()[
                    0
                ].floor_thickness_meters,
                0.3,
            )
            self.assertTrue(
                self.workspace._canvas_surface_mesh_update_timer.isActive()
            )
            self.assertTrue(
                self.workspace._pending_canvas_surface_mesh_update
            )
            schedule_refresh.assert_not_called()
            reconcile_assignments.assert_not_called()

            self.workspace._commit_pending_canvas_surface_mesh_update()

        reconcile_assignments.assert_called_once_with([level])
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertAlmostEqual(
            self.workspace._build_viewer_preview_levels()[
                0
            ].floor_thickness_meters,
            0.7,
        )

    def test_floor_spinbox_uses_the_same_delayed_mesh_commit(self) -> None:
        level, _floor, _target = self._install_level()
        revision_before = self.workspace._viewer_preview_revision

        with (
            patch.object(
                self.workspace,
                "_schedule_viewer_preview_refresh",
            ) as schedule_refresh,
            patch.object(
                self.workspace.surface_texture_generation,
                "reconcile_assignments_with_levels",
                return_value=False,
            ) as reconcile_assignments,
        ):
            self.workspace.floor_thickness_spinbox.setValue(0.65)

            self.assertAlmostEqual(level.floor_thickness_meters, 0.65)
            self.assertAlmostEqual(
                self.workspace._build_viewer_preview_levels()[
                    0
                ].floor_thickness_meters,
                0.3,
            )
            self.assertTrue(
                self.workspace._canvas_surface_mesh_update_timer.isActive()
            )
            self.assertTrue(
                self.workspace._pending_canvas_surface_mesh_update
            )
            self.assertEqual(
                self.workspace._viewer_preview_revision,
                revision_before,
            )
            schedule_refresh.assert_not_called()
            reconcile_assignments.assert_not_called()

            self.workspace._commit_pending_canvas_surface_mesh_update()

        reconcile_assignments.assert_called_once_with([level])
        schedule_refresh.assert_called_once_with(preserve_camera=True)
        self.assertAlmostEqual(
            self.workspace._build_viewer_preview_levels()[
                0
            ].floor_thickness_meters,
            0.65,
        )
        self.assertFalse(
            self.workspace._canvas_surface_mesh_update_timer.isActive()
        )
        self.assertFalse(self.workspace._pending_canvas_surface_mesh_update)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
