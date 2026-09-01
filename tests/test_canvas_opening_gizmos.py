# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication
import trimesh

from housemaker.canvas_openings import (
    CANVAS_OPENING_DOORWAY,
    CANVAS_OPENING_WINDOW,
    CanvasOpeningBounds,
    CanvasOpeningReference,
    CanvasOpeningTarget,
)
from housemaker.glb import GeneratedModel
from housemaker.surface_geometry import FixedSurface, SURFACE_TYPE_WALL
from housemaker.viewer import (
    CANVAS_OPENING_GIZMO_ANCHOR,
    CANVAS_OPENING_GIZMO_SIDE,
    CANVAS_OPENING_HANDLE_HIT_RADIUS_PIXELS,
    CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
    CANVAS_OPENING_SIDE_BOTTOM,
    CANVAS_OPENING_SIDE_LEFT,
    CANVAS_OPENING_SIDE_RIGHT,
    CANVAS_OPENING_SIDE_SIZE_PIXELS,
    CANVAS_OPENING_SIDE_TOP,
    GlbViewerWidget,
    _CanvasOpeningGizmoHandle,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_target(
    kind: str = CANVAS_OPENING_WINDOW,
    *,
    item_index: int = 0,
    stable_id: str = "window-a",
    bounds: CanvasOpeningBounds | None = None,
) -> CanvasOpeningTarget:
    return CanvasOpeningTarget(
        reference=CanvasOpeningReference(
            kind=kind,
            level_index=0,
            item_index=item_index,
            stable_id=stable_id if kind == CANVAS_OPENING_WINDOW else None,
        ),
        wall_surface_id="level:0/wall:1:2",
        plane_start_world=(0.0, 0.0, 0.0),
        wall_tangent_world=(1.0, 0.0, 0.0),
        wall_normal_world=(0.0, -1.0, 0.0),
        wall_width_meters=4.0,
        wall_height_meters=3.0,
        minimum_width_meters=0.2,
        minimum_height_meters=0.3,
        bounds=bounds
        or CanvasOpeningBounds(
            start_ratio=0.25,
            end_ratio=0.75,
            bottom_ratio=0.2,
            top_ratio=0.8,
        ),
    )


def _build_wall(*, y_offset: float = 0.0) -> FixedSurface:
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
        surface_id="level:0/wall:1:2",
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


def _build_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(4.0, 0.2, 3.0))
    mesh.apply_translation((2.0, 0.0, 1.5))
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


def _ray_at(x: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((x, -2.0, z), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


def _bounds_values(bounds: CanvasOpeningBounds) -> tuple[float, ...]:
    return (
        bounds.start_ratio,
        bounds.end_ratio,
        bounds.bottom_ratio,
        bounds.top_ratio,
    )


# ### Canvas opening gizmo tests ###
class CanvasOpeningGizmoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.exit_first_person_mode()
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(self, target: CanvasOpeningTarget) -> GlbViewerWidget:
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((_build_wall(),))
        viewer.set_canvas_opening_targets((target,))
        self.widgets.append(viewer)
        return viewer

    def test_hole_plane_selection_wins_a_coplanar_wall_pick(self) -> None:
        target = _build_target()
        viewer = self._build_viewer(target)
        selected: list[object] = []
        viewer.canvas_opening_selection_changed.connect(selected.append)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_window_wall_pick_requested(QPointF(20.0, 20.0))

        self.assertEqual(
            viewer.get_selected_canvas_opening_reference(),
            target.reference,
        )
        self.assertIsNone(viewer.get_selected_wall_surface_id())
        self.assertEqual(selected, [target.reference])

    def test_opening_and_handles_remain_selectable_through_a_nearer_wall(
        self,
    ) -> None:
        target = _build_target()
        viewer = GlbViewerWidget(window_editing_enabled=True)
        viewer.set_wall_targets((_build_wall(y_offset=-1.0),))
        viewer.set_canvas_opening_targets((target,))
        self.widgets.append(viewer)

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._handle_window_wall_pick_requested(QPointF())

        self.assertEqual(
            viewer.get_selected_canvas_opening_reference(),
            target.reference,
        )
        self.assertIsNone(viewer.get_selected_wall_surface_id())

        with (
            patch.object(viewer.view, "pixelSize", return_value=0.01),
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=_ray_at(1.0, 1.5),
            ),
        ):
            viewer._handle_placed_object_pointer_pressed(QPointF())

        self.assertIsNotNone(viewer._canvas_opening_edit_drag)
        assert viewer._canvas_opening_edit_drag is not None
        self.assertEqual(
            viewer._canvas_opening_edit_drag.handle,
            _CanvasOpeningGizmoHandle(
                CANVAS_OPENING_GIZMO_SIDE,
                CANVAS_OPENING_SIDE_LEFT,
            ),
        )
        self.assertTrue(viewer.view.is_primary_pointer_drag_reserved)
        viewer._cancel_canvas_opening_edit_drag()

    def test_selected_opening_renders_handles_and_survives_model(
        self,
    ) -> None:
        target = _build_target()
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)
        first_items = tuple(viewer._canvas_opening_gizmo_items)

        self.assertEqual(len(first_items), 3)
        for item in first_items:
            self.assertEqual(
                item.depthValue(),
                CANVAS_OPENING_OVERLAY_DEPTH_VALUE,
            )
            gl_options = getattr(item, "_GLGraphicsItem__glOpts")
            self.assertFalse(gl_options[GL.GL_DEPTH_TEST])
            self.assertTrue(gl_options[GL.GL_BLEND])
        side_positions = np.asarray(first_items[1].pos, dtype=float)
        anchor_positions = np.asarray(first_items[2].pos, dtype=float)
        self.assertEqual(side_positions.shape, (4, 3))
        self.assertEqual(anchor_positions.shape, (1, 3))
        np.testing.assert_allclose(
            side_positions[:, (0, 2)],
            ((1.0, 1.5), (3.0, 1.5), (2.0, 0.6), (2.0, 2.4)),
            atol=1e-9,
        )
        self.assertGreaterEqual(CANVAS_OPENING_SIDE_SIZE_PIXELS, 20.0)
        self.assertGreaterEqual(CANVAS_OPENING_HANDLE_HIT_RADIUS_PIXELS, 18.0)

        viewer.set_model(_build_model(), preserve_camera=True)

        self.assertEqual(
            viewer.get_selected_canvas_opening_reference(),
            target.reference,
        )
        self.assertEqual(len(viewer._canvas_opening_gizmo_items), 3)
        self.assertTrue(
            first_items[0] not in viewer._canvas_opening_gizmo_items
        )

    def test_cpu_handle_pick_distinguishes_each_side_from_the_anchor(self) -> None:
        target = _build_target()
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)

        with patch.object(viewer.view, "pixelSize", return_value=0.01):
            side_handles = tuple(
                viewer._pick_canvas_opening_gizmo_handle(*_ray_at(x, z))
                for x, z in (
                    (1.0, 1.5),
                    (3.0, 1.5),
                    (2.0, 0.6),
                    (2.0, 2.4),
                )
            )
            anchor_handle = viewer._pick_canvas_opening_gizmo_handle(
                *_ray_at(2.0, 1.5)
            )

        self.assertEqual(
            side_handles,
            tuple(
                _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_SIDE, side)
                for side in (
                    CANVAS_OPENING_SIDE_LEFT,
                    CANVAS_OPENING_SIDE_RIGHT,
                    CANVAS_OPENING_SIDE_BOTTOM,
                    CANVAS_OPENING_SIDE_TOP,
                )
            ),
        )
        self.assertEqual(
            anchor_handle,
            _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_ANCHOR),
        )

    def test_each_side_drag_changes_only_its_own_boundary(self) -> None:
        cases = (
            (
                CANVAS_OPENING_SIDE_LEFT,
                (1.0, 1.5),
                (0.4, 0.3),
                CanvasOpeningBounds(0.1, 0.75, 0.2, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_RIGHT,
                (3.0, 1.5),
                (3.6, 0.3),
                CanvasOpeningBounds(0.25, 0.9, 0.2, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_BOTTOM,
                (2.0, 0.6),
                (0.4, 0.3),
                CanvasOpeningBounds(0.25, 0.75, 0.1, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_TOP,
                (2.0, 2.4),
                (0.4, 2.7),
                CanvasOpeningBounds(0.25, 0.75, 0.2, 0.9),
            ),
        )
        for side, start_point, finish_point, expected_bounds in cases:
            with self.subTest(side=side):
                target = _build_target()
                viewer = self._build_viewer(target)
                viewer.select_canvas_opening(target.reference)
                started: list[object] = []
                previews: list[object] = []
                finished: list[tuple[object, bool]] = []
                viewer.canvas_opening_edit_started.connect(started.append)
                viewer.canvas_opening_edit_preview_changed.connect(previews.append)
                viewer.canvas_opening_edit_finished.connect(
                    lambda edit, changed: finished.append((edit, changed))
                )
                handle = _CanvasOpeningGizmoHandle(
                    CANVAS_OPENING_GIZMO_SIDE,
                    side,
                )

                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(*start_point),
                ):
                    self.assertTrue(
                        viewer._begin_canvas_opening_gizmo_drag(handle, QPointF())
                    )
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(*finish_point),
                ):
                    self.assertTrue(
                        viewer._finish_canvas_opening_gizmo_drag(QPointF())
                    )

                self.assertEqual(len(started), 1)
                self.assertEqual(len(previews), 1)
                self.assertEqual(len(finished), 1)
                self.assertTrue(finished[0][1])
                np.testing.assert_allclose(
                    _bounds_values(finished[0][0].bounds),
                    _bounds_values(expected_bounds),
                    atol=1e-9,
                )
                self.assertEqual(
                    viewer._canvas_opening_targets[target.key].bounds,
                    finished[0][0].bounds,
                )
                self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)

    def test_window_anchor_moves_both_axes_and_doorway_only_sideways(
        self,
    ) -> None:
        handle = _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_ANCHOR)
        for kind, expected_bounds in (
            (
                CANVAS_OPENING_WINDOW,
                CanvasOpeningBounds(0.375, 0.875, 0.3, 0.9),
            ),
            (
                CANVAS_OPENING_DOORWAY,
                CanvasOpeningBounds(0.375, 0.875, 0.2, 0.8),
            ),
        ):
            with self.subTest(kind=kind):
                target = _build_target(kind)
                viewer = self._build_viewer(target)
                viewer.select_canvas_opening(target.reference)
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(2.0, 1.5),
                ):
                    viewer._begin_canvas_opening_gizmo_drag(handle, QPointF())
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(2.5, 1.8),
                ):
                    viewer._finish_canvas_opening_gizmo_drag(QPointF())

                np.testing.assert_allclose(
                    _bounds_values(
                        viewer._canvas_opening_targets[target.key].bounds
                    ),
                    _bounds_values(expected_bounds),
                    atol=1e-9,
                )

    def test_side_drags_clamp_to_wall_and_minimum_size(self) -> None:
        cases = (
            (
                CANVAS_OPENING_SIDE_LEFT,
                (1.0, 1.5),
                (20.0, 20.0),
                CanvasOpeningBounds(0.7, 0.75, 0.2, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_RIGHT,
                (3.0, 1.5),
                (-20.0, -20.0),
                CanvasOpeningBounds(0.25, 0.3, 0.2, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_BOTTOM,
                (2.0, 0.6),
                (20.0, 20.0),
                CanvasOpeningBounds(0.25, 0.75, 0.7, 0.8),
            ),
            (
                CANVAS_OPENING_SIDE_TOP,
                (2.0, 2.4),
                (-20.0, -20.0),
                CanvasOpeningBounds(0.25, 0.75, 0.2, 0.3),
            ),
        )
        for side, start_point, finish_point, expected_bounds in cases:
            with self.subTest(side=side):
                target = _build_target()
                viewer = self._build_viewer(target)
                viewer.select_canvas_opening(target.reference)
                handle = _CanvasOpeningGizmoHandle(
                    CANVAS_OPENING_GIZMO_SIDE,
                    side,
                )
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(*start_point),
                ):
                    viewer._begin_canvas_opening_gizmo_drag(handle, QPointF())
                with patch.object(
                    viewer.view,
                    "build_camera_ray",
                    return_value=_ray_at(*finish_point),
                ):
                    viewer._finish_canvas_opening_gizmo_drag(QPointF())

                np.testing.assert_allclose(
                    _bounds_values(
                        viewer._canvas_opening_targets[target.key].bounds
                    ),
                    _bounds_values(expected_bounds),
                    atol=1e-9,
                )

    def test_escape_restores_start_and_delete_is_consumed(self) -> None:
        target = _build_target()
        viewer = self._build_viewer(target)
        viewer.select_canvas_opening(target.reference)
        cancelled: list[object] = []
        finished: list[object] = []
        generic_deletions: list[bool] = []
        viewer.canvas_opening_edit_cancelled.connect(cancelled.append)
        viewer.canvas_opening_edit_finished.connect(
            lambda *_args: finished.append(True)
        )
        viewer.delete_requested.connect(lambda: generic_deletions.append(True))
        handle = _CanvasOpeningGizmoHandle(CANVAS_OPENING_GIZMO_ANCHOR)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.5),
        ):
            viewer._begin_canvas_opening_gizmo_drag(handle, QPointF())
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.5, 1.8),
        ):
            viewer._update_canvas_opening_gizmo_drag(QPointF())

        viewer.view.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        viewer.view.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Delete,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0].bounds, target.bounds)
        self.assertEqual(finished, [])
        self.assertEqual(generic_deletions, [])
        self.assertEqual(viewer._canvas_opening_targets[target.key], target)
        self.assertFalse(viewer.view.is_primary_pointer_drag_reserved)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
