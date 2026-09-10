# ### Environment setup ###
from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
import trimesh

from housemaker.canvas_surface_edits import (
    CANVAS_SURFACE_EDIT_LEVEL_HEIGHT,
    CANVAS_SURFACE_EDIT_WALL_VERTEX,
    CanvasSurfaceEditHandleTarget,
    CanvasSurfaceEditReference,
)
from housemaker.glb import GeneratedModel
from housemaker.surface_geometry import (
    SURFACE_TYPE_CEILING,
    SURFACE_TYPE_WALL,
    FixedSurface,
)
from housemaker.viewer import (
    CANVAS_OPENING_GIZMO_ANCHOR,
    CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
    CANVAS_SURFACE_EDIT_GIZMO_ENDPOINT_SIZE_PIXELS,
    CANVAS_SURFACE_EDIT_GIZMO_LINE_WIDTH,
    GlbViewerWidget,
    _CanvasOpeningGizmoHandle,
    _TransformGizmoHandle,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_wall(
    surface_id: str = "level:0/wall:1:2",
    *,
    y_offset: float = 0.0,
) -> FixedSurface:
    vertices = np.asarray(
        (
            (0.0, y_offset, 0.0),
            (4.0, y_offset, 0.0),
            (4.0, y_offset, 3.0),
            (0.0, y_offset, 3.0),
        ),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=SURFACE_TYPE_WALL,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
        wall_key="1:2",
        wall_start_world=(0.0, y_offset, 0.0),
        wall_end_world=(4.0, y_offset, 0.0),
        wall_height_meters=3.0,
    )


def _build_horizontal_surface(
    surface_type: str,
    *,
    surface_id: str,
    z: float,
) -> FixedSurface:
    vertices = np.asarray(
        ((0.0, 0.0, z), (4.0, 0.0, z), (4.0, 3.0, z), (0.0, 3.0, z)),
        dtype=float,
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id=surface_id,
        surface_type=surface_type,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=12.0,
    )


def _build_wall_targets(
    surface: FixedSurface,
) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
    targets: list[CanvasSurfaceEditHandleTarget] = []
    for vertex_id, origin in (
        (1, surface.wall_start_world),
        (2, surface.wall_end_world),
    ):
        assert origin is not None
        for axis_index in (0, 1):
            targets.append(
                CanvasSurfaceEditHandleTarget(
                    reference=CanvasSurfaceEditReference(
                        kind=CANVAS_SURFACE_EDIT_WALL_VERTEX,
                        level_index=0,
                        axis_index=axis_index,
                        vertex_id=vertex_id,
                    ),
                    surface_id=surface.surface_id,
                    origin_world=origin,
                    axis_world=tuple(np.eye(3, dtype=float)[axis_index]),
                    minimum_delta_meters=-0.5,
                    maximum_delta_meters=0.75,
                    chain_vertex_ids=(1, 2),
                    chain_vertex_ratios=(0.0, 1.0),
                    chain_vertex_image_positions=((0.0, 0.0), (4.0, 0.0)),
                )
            )
    return tuple(targets)


def _build_vertical_target(
    surface: FixedSurface,
) -> CanvasSurfaceEditHandleTarget:
    origin = np.asarray(surface.mesh.centroid, dtype=float).copy()
    return CanvasSurfaceEditHandleTarget(
        reference=CanvasSurfaceEditReference(
            kind=CANVAS_SURFACE_EDIT_LEVEL_HEIGHT,
            level_index=0,
            axis_index=2,
        ),
        surface_id=surface.surface_id,
        origin_world=tuple(float(value) for value in origin),
        axis_world=(0.0, 0.0, 1.0),
        minimum_delta_meters=-0.5,
        maximum_delta_meters=0.75,
        baseline_value_meters=3.0,
    )


def _move_first_wall_endpoint_targets(
    targets: tuple[CanvasSurfaceEditHandleTarget, ...],
    x_position: float,
) -> tuple[CanvasSurfaceEditHandleTarget, ...]:
    """Rebase one fixture wall's handles after moving its first endpoint."""

    chain_positions = ((float(x_position), 0.0), (4.0, 0.0))
    return tuple(
        replace(
            target,
            origin_world=(
                float(x_position),
                target.origin_world[1],
                target.origin_world[2],
            ),
            chain_vertex_image_positions=chain_positions,
        )
        if target.reference.vertex_id == 1
        else replace(
            target,
            chain_vertex_image_positions=chain_positions,
        )
        for target in targets
    )


def _build_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(4.0, 3.0, 3.0))
    mesh.apply_translation((2.0, 1.5, 1.5))
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


def _ray_at_x(x: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((x, -2.0, 0.0), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


# ### Canvas structural gizmo tests ###
class CanvasSurfaceEditGizmoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(
        self,
        surfaces: tuple[FixedSurface, ...],
        targets: tuple[CanvasSurfaceEditHandleTarget, ...],
    ) -> GlbViewerWidget:
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets(surfaces)
        viewer.set_canvas_surface_edit_targets(targets)
        self.widgets.append(viewer)
        return viewer

    def test_last_picked_surface_is_distinct_from_multi_selection(self) -> None:
        first = _build_wall()
        second = _build_wall("level:0/room:7/wall:1:2", y_offset=1.0)
        targets = (*_build_wall_targets(first), *_build_wall_targets(second))
        viewer = self._build_viewer((first, second), targets)
        emitted: list[object] = []
        viewer.canvas_surface_selection_changed.connect(emitted.append)

        viewer.select_canvas_surface_target(first.surface_id)
        viewer.select_canvas_surface_target(second.surface_id, additive=True)

        self.assertEqual(
            viewer.get_selected_canvas_surface_ids(),
            (first.surface_id, second.surface_id),
        )
        self.assertEqual(viewer.get_active_canvas_surface_id(), second.surface_id)
        self.assertTrue(
            all(
                target.surface_id == second.surface_id
                for target in viewer._get_active_canvas_surface_edit_targets()
            )
        )

        selection_event_count = len(emitted)
        viewer.set_canvas_surface_edit_targets(tuple(reversed(targets)))

        self.assertEqual(viewer.get_active_canvas_surface_id(), second.surface_id)
        self.assertEqual(len(emitted), selection_event_count)

    def test_wall_renders_four_prominent_depth_free_axis_handles(self) -> None:
        wall = _build_wall()
        targets = _build_wall_targets(wall)
        viewer = self._build_viewer((wall,), targets)
        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.select_canvas_surface_target(wall.surface_id)
            viewer._refresh_canvas_surface_edit_gizmo_items()

        items = tuple(viewer._canvas_surface_edit_gizmo_items)
        self.assertEqual(len(items), 9)
        for item in items:
            self.assertEqual(item.depthValue(), CANVAS_OPENING_OVERLAY_DEPTH_VALUE)
            gl_options = getattr(item, "_GLGraphicsItem__glOpts")
            self.assertFalse(gl_options[GL.GL_DEPTH_TEST])
        axis_items = items[1::2]
        self.assertEqual(len(axis_items), 4)
        self.assertGreaterEqual(CANVAS_SURFACE_EDIT_GIZMO_LINE_WIDTH, 4.0)
        self.assertGreaterEqual(
            CANVAS_SURFACE_EDIT_GIZMO_ENDPOINT_SIZE_PIXELS,
            14.0,
        )

    def test_ceiling_handle_points_up(self) -> None:
        ceiling = _build_horizontal_surface(
            SURFACE_TYPE_CEILING,
            surface_id="level:0/ceiling",
            z=3.0,
        )
        target = _build_vertical_target(ceiling)
        viewer = self._build_viewer((ceiling,), (target,))

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.select_canvas_surface_target(ceiling.surface_id)
            viewer._refresh_canvas_surface_edit_gizmo_items()
        axis_positions = np.asarray(
            viewer._canvas_surface_edit_gizmo_items[1].pos,
            dtype=float,
        )
        delta = axis_positions[1] - axis_positions[0]
        self.assertAlmostEqual(delta[0], 0.0)
        self.assertAlmostEqual(delta[1], 0.0)
        self.assertGreater(delta[2], 0.0)

    def test_cpu_segment_pick_returns_the_matching_wall_axis(self) -> None:
        wall = _build_wall()
        targets = _build_wall_targets(wall)
        viewer = self._build_viewer((wall,), targets)
        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            viewer.select_canvas_surface_target(wall.surface_id)
            viewer._refresh_canvas_surface_edit_gizmo_items()
            picked = viewer._pick_canvas_surface_edit_handle(*_ray_at_x(0.5))

        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.reference.vertex_id, 1)
        self.assertEqual(picked.reference.axis_index, 0)

    def test_pointer_priority_is_opening_then_structural_then_object(self) -> None:
        wall = _build_wall()
        structural_target = _build_wall_targets(wall)[0]
        viewer = self._build_viewer((wall,), (structural_target,))
        opening_handle = _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_ANCHOR)
        object_handle = _TransformGizmoHandle("translate", 0)
        camera_ray = _ray_at_x(0.5)

        with (
            patch.object(viewer.view, "build_camera_ray", return_value=camera_ray),
            patch.object(
                viewer,
                "_pick_canvas_opening_gizmo_handle",
                return_value=opening_handle,
            ),
            patch.object(viewer, "_begin_canvas_opening_gizmo_drag") as begin_opening,
            patch.object(viewer, "_begin_canvas_surface_edit_drag") as begin_surface,
            patch.object(viewer, "_begin_placed_object_gizmo_drag") as begin_object,
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF())
            begin_opening.assert_called_once_with(opening_handle, QPointF())
            begin_surface.assert_not_called()
            begin_object.assert_not_called()

        with (
            patch.object(viewer.view, "build_camera_ray", return_value=camera_ray),
            patch.object(
                viewer,
                "_pick_canvas_opening_gizmo_handle",
                return_value=None,
            ),
            patch.object(
                viewer,
                "_pick_canvas_surface_edit_handle",
                return_value=structural_target,
            ),
            patch.object(
                viewer,
                "_pick_transform_gizmo_handle",
                return_value=object_handle,
            ),
            patch.object(viewer, "_begin_canvas_surface_edit_drag") as begin_surface,
            patch.object(viewer, "_begin_placed_object_gizmo_drag") as begin_object,
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF())
            begin_surface.assert_called_once_with(structural_target, QPointF())
            begin_object.assert_not_called()

    def test_drag_emits_absolute_deltas_and_moves_only_live_overlays(self) -> None:
        wall = _build_wall()
        targets = _build_wall_targets(wall)
        target = targets[0]
        viewer = self._build_viewer((wall,), targets)
        viewer.select_canvas_surface_target(wall.surface_id)
        started: list[object] = []
        previews: list[object] = []
        finished: list[tuple[object, bool]] = []
        viewer.canvas_surface_edit_started.connect(started.append)
        viewer.canvas_surface_edit_preview_changed.connect(previews.append)
        viewer.canvas_surface_edit_finished.connect(
            lambda edit, changed: finished.append((edit, changed))
        )

        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(viewer.view, "build_camera_ray", return_value=_ray_at_x(0.0)),
        ):
            self.assertTrue(
                viewer._begin_canvas_surface_edit_drag(target, QPointF())
            )
        self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)

        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(viewer.view, "build_camera_ray", return_value=_ray_at_x(0.4)),
        ):
            self.assertTrue(viewer._update_canvas_surface_edit_drag(QPointF()))
        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(viewer.view, "build_camera_ray", return_value=_ray_at_x(0.6)),
        ):
            self.assertTrue(viewer._update_canvas_surface_edit_drag(QPointF()))
            shared_endpoint_origins = tuple(
                np.asarray(viewer._canvas_surface_edit_gizmo_items[index].pos)[0]
                for index in (1, 3)
            )
            self.assertTrue(viewer._finish_canvas_surface_edit_drag(QPointF()))

        self.assertEqual([edit.delta_meters for edit in started], [0.0])
        self.assertEqual([edit.delta_meters for edit in previews], [0.4, 0.6])
        self.assertEqual(len(finished), 1)
        self.assertAlmostEqual(finished[0][0].delta_meters, 0.6)
        self.assertTrue(finished[0][1])
        for origin in shared_endpoint_origins:
            np.testing.assert_allclose(origin, (0.6, 0.0, 0.0), atol=1e-9)
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_changed_release_keeps_passive_outline_until_mesh_refresh(
        self,
    ) -> None:
        wall = _build_wall()
        targets = _build_wall_targets(wall)
        target = targets[0]
        viewer = self._build_viewer((wall,), targets)
        viewer.select_canvas_surface_target(wall.surface_id)
        viewer.canvas_surface_edit_finished.connect(
            lambda _edit, _changed: viewer.set_canvas_surface_edit_targets(
                (),
                preserve_pending_outline=True,
            )
        )

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(0.0),
        ):
            self.assertTrue(
                viewer._begin_canvas_surface_edit_drag(target, QPointF())
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(0.5),
        ):
            self.assertTrue(viewer._finish_canvas_surface_edit_drag(QPointF()))

        pending_outline = viewer._canvas_surface_edit_pending_outline_positions
        self.assertIsNotNone(pending_outline)
        assert pending_outline is not None
        self.assertEqual(len(viewer._canvas_surface_edit_gizmo_items), 1)
        np.testing.assert_allclose(
            np.asarray(viewer._canvas_surface_edit_gizmo_items[0].pos),
            pending_outline,
            atol=1e-9,
        )

        viewer.clear_canvas_surface_edit_pending_outline()

        self.assertIsNone(viewer._canvas_surface_edit_pending_outline_positions)
        self.assertEqual(viewer._canvas_surface_edit_gizmo_items, [])

    def test_empty_target_replacement_clears_pending_outline_by_default(
        self,
    ) -> None:
        wall = _build_wall()
        targets = _build_wall_targets(wall)
        viewer = self._build_viewer((wall,), targets)
        viewer.select_canvas_surface_target(wall.surface_id)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(0.0),
        ):
            self.assertTrue(
                viewer._begin_canvas_surface_edit_drag(targets[0], QPointF())
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(0.5),
        ):
            self.assertTrue(viewer._finish_canvas_surface_edit_drag(QPointF()))

        self.assertIsNotNone(
            viewer._canvas_surface_edit_pending_outline_positions
        )
        viewer.set_canvas_surface_edit_targets(())

        self.assertIsNone(viewer._canvas_surface_edit_pending_outline_positions)
        self.assertIsNone(viewer._canvas_surface_edit_pending_outline_surface_id)
        self.assertEqual(viewer._canvas_surface_edit_gizmo_items, [])

    def test_rebased_handles_remain_visible_and_pickable_after_release(
        self,
    ) -> None:
        wall = _build_wall()
        targets = _build_wall_targets(wall)
        target = targets[0]
        viewer = self._build_viewer((wall,), targets)
        viewer.select_canvas_surface_target(wall.surface_id)

        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at_x(0.0),
            ),
        ):
            self.assertTrue(
                viewer._begin_canvas_surface_edit_drag(target, QPointF())
            )
        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at_x(0.5),
            ),
        ):
            self.assertTrue(viewer._finish_canvas_surface_edit_drag(QPointF()))
            rebased_targets = _move_first_wall_endpoint_targets(targets, 0.5)
            viewer.set_canvas_surface_edit_targets(
                rebased_targets,
                preserve_pending_outline=True,
            )
            picked = viewer._pick_canvas_surface_edit_handle(*_ray_at_x(0.75))

        self.assertEqual(len(viewer._canvas_surface_edit_gizmo_items), 9)
        self.assertIsNotNone(picked)
        assert picked is not None
        self.assertEqual(picked.reference.vertex_id, 1)
        self.assertEqual(picked.reference.axis_index, 0)
        self.assertAlmostEqual(picked.origin_world[0], 0.5)

    def test_follow_up_drag_uses_and_restores_pending_outline_baseline(
        self,
    ) -> None:
        wall = _build_wall()
        targets = _build_wall_targets(wall)
        viewer = self._build_viewer((wall,), targets)
        viewer.select_canvas_surface_target(wall.surface_id)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(0.0),
        ):
            self.assertTrue(
                viewer._begin_canvas_surface_edit_drag(targets[0], QPointF())
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(0.5),
        ):
            self.assertTrue(viewer._finish_canvas_surface_edit_drag(QPointF()))

        rebased_targets = _move_first_wall_endpoint_targets(targets, 0.5)
        viewer.set_canvas_surface_edit_targets(
            rebased_targets,
            preserve_pending_outline=True,
        )
        first_pending_outline = np.asarray(
            viewer._canvas_surface_edit_pending_outline_positions,
            dtype=float,
        ).copy()

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(0.5),
        ):
            self.assertTrue(
                viewer._begin_canvas_surface_edit_drag(
                    rebased_targets[0],
                    QPointF(),
                )
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(0.75),
        ):
            self.assertTrue(viewer._update_canvas_surface_edit_drag(QPointF()))
            live_outline = np.asarray(
                viewer._canvas_surface_edit_gizmo_items[0].pos,
                dtype=float,
            )
            self.assertAlmostEqual(float(np.min(live_outline[:, 0])), 0.75)
            self.assertTrue(viewer._finish_canvas_surface_edit_drag(QPointF()))

        second_pending_outline = np.asarray(
            viewer._canvas_surface_edit_pending_outline_positions,
            dtype=float,
        ).copy()
        self.assertAlmostEqual(
            float(np.min(second_pending_outline[:, 0])),
            0.75,
        )

        twice_rebased_targets = _move_first_wall_endpoint_targets(targets, 0.75)
        viewer.set_canvas_surface_edit_targets(
            twice_rebased_targets,
            preserve_pending_outline=True,
        )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(0.75),
        ):
            self.assertTrue(
                viewer._begin_canvas_surface_edit_drag(
                    twice_rebased_targets[0],
                    QPointF(),
                )
            )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at_x(1.0),
        ):
            self.assertTrue(viewer._update_canvas_surface_edit_drag(QPointF()))
        self.assertTrue(viewer.cancel_canvas_surface_edit())

        np.testing.assert_allclose(
            viewer._canvas_surface_edit_pending_outline_positions,
            second_pending_outline,
            atol=1e-9,
        )
        self.assertFalse(
            np.allclose(first_pending_outline, second_pending_outline)
        )
        self.assertEqual(len(viewer._canvas_surface_edit_gizmo_items), 9)

    def test_context_changes_cancel_drag_and_model_refresh_keeps_active_surface(
        self,
    ) -> None:
        wall = _build_wall()
        targets = _build_wall_targets(wall)
        target = targets[0]
        viewer = self._build_viewer((wall,), targets)
        viewer.select_canvas_surface_target(wall.surface_id)
        cancelled: list[object] = []
        viewer.canvas_surface_edit_cancelled.connect(cancelled.append)

        def begin_drag() -> None:
            with patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at_x(0.0),
            ):
                self.assertTrue(
                    viewer._begin_canvas_surface_edit_drag(target, QPointF())
                )

        begin_drag()
        viewer.set_canvas_surface_edit_targets(targets)
        begin_drag()
        viewer.select_canvas_surface_target(None)
        viewer.select_canvas_surface_target(wall.surface_id)

        first_items = tuple(viewer._canvas_surface_edit_gizmo_items)
        begin_drag()
        viewer.set_model(_build_model(), preserve_camera=True)

        self.assertEqual(viewer.get_active_canvas_surface_id(), wall.surface_id)
        self.assertEqual(len(viewer._canvas_surface_edit_gizmo_items), 9)
        self.assertNotIn(first_items[0], viewer._canvas_surface_edit_gizmo_items)

        begin_drag()
        viewer.navigation_mode_changed.emit("first_person")

        self.assertEqual(
            [edit.delta_meters for edit in cancelled],
            [0.0, 0.0, 0.0, 0.0],
        )
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)
        self.assertIsNone(viewer._canvas_surface_edit_drag)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
