# ### Environment setup ###
from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication
import trimesh

from housemaker.glb import GeneratedModel
from housemaker.surface_geometry import (
    FixedSurface,
    SURFACE_TYPE_FLOOR,
    SURFACE_TYPE_WALL,
    WallWindowPlacement,
)
from housemaker.viewer import GlbViewerWidget


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_wall(
    *,
    surface_id: str = "level:0/wall:1:2",
    x_offset: float = 0.0,
) -> FixedSurface:
    vertices = np.asarray(
        (
            (x_offset, 0.0, 0.0),
            (x_offset + 4.0, 0.0, 0.0),
            (x_offset + 4.0, 0.0, 3.0),
            (x_offset, 0.0, 3.0),
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
        wall_start_world=(x_offset, 0.0, 0.0),
        wall_end_world=(x_offset + 4.0, 0.0, 0.0),
        wall_height_meters=3.0,
    )


def _build_floor() -> FixedSurface:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(
            ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, 4.0, 0.0)),
            dtype=float,
        ),
        faces=np.asarray(((0, 1, 2),), dtype=np.int64),
        process=False,
    )
    return FixedSurface(
        surface_id="level:0/floor",
        surface_type=SURFACE_TYPE_FLOOR,
        level_index=0,
        room_index=None,
        mesh=mesh,
        area_square_meters=8.0,
    )


def _build_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(4.0, 0.2, 3.0))
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


def _ray_at(x: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray((x, -1.0, z), dtype=float),
        np.asarray((0.0, 1.0, 0.0), dtype=float),
    )


# ### Canvas window editor tests ###
class CanvasWindowEditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[GlbViewerWidget] = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        _qt_application.processEvents()

    def _build_viewer(self, *, enabled: bool = True) -> GlbViewerWidget:
        viewer = GlbViewerWidget(window_editing_enabled=enabled)
        self.widgets.append(viewer)
        return viewer

    def _arm_selected_wall(self, viewer: GlbViewerWidget) -> FixedSurface:
        wall = _build_wall()
        viewer.set_wall_targets((wall,))
        self.assertTrue(viewer.select_wall_target(wall.surface_id))
        self.assertTrue(viewer.begin_window_placement())
        return wall

    def _build_valid_preview(self, viewer: GlbViewerWidget) -> None:
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(1.0, 0.5),
        ):
            viewer._handle_window_pointer_pressed(QPointF(10.0, 10.0))
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(3.0, 2.0),
        ):
            viewer._handle_window_pointer_moved(QPointF(40.0, 40.0))

    def test_window_tools_are_canvas_opt_in_and_require_one_wall(self) -> None:
        ordinary_viewer = self._build_viewer(enabled=False)
        canvas_viewer = self._build_viewer()

        self.assertIsNone(ordinary_viewer.window_tools_panel)
        self.assertIsNone(ordinary_viewer.add_window_button)
        self.assertIsNone(ordinary_viewer.undo_window_button)
        self.assertIsNotNone(canvas_viewer.window_tools_panel)
        self.assertIsNotNone(canvas_viewer.add_window_button)
        self.assertIsNotNone(canvas_viewer.undo_window_button)
        assert canvas_viewer.add_window_button is not None
        assert canvas_viewer.undo_window_button is not None
        self.assertEqual(canvas_viewer.add_window_button.text(), "Add window")
        self.assertEqual(canvas_viewer.undo_window_button.text(), "Undo window")
        self.assertFalse(canvas_viewer.add_window_button.isEnabled())
        self.assertFalse(canvas_viewer.undo_window_button.isEnabled())

        panel_layout = canvas_viewer.window_tools_panel.layout()
        self.assertIs(
            panel_layout.itemAt(1).widget(),
            canvas_viewer.add_window_button,
        )
        self.assertIs(
            panel_layout.itemAt(2).widget(),
            canvas_viewer.undo_window_button,
        )

        wall = _build_wall()
        canvas_viewer.set_wall_targets((_build_floor(), wall))

        self.assertFalse(canvas_viewer.add_window_button.isEnabled())
        self.assertTrue(canvas_viewer.select_wall_target(wall.surface_id))
        self.assertTrue(canvas_viewer.add_window_button.isEnabled())

    def test_undo_availability_is_independent_of_selection_and_pauses_drawing(
        self,
    ) -> None:
        viewer = self._build_viewer()
        assert viewer.undo_window_button is not None
        emitted: list[bool] = []
        viewer.window_undo_requested.connect(lambda: emitted.append(True))

        viewer.set_window_undo_available(True)

        self.assertTrue(viewer.undo_window_button.isEnabled())
        wall = self._arm_selected_wall(viewer)
        self.assertEqual(viewer.get_selected_wall_surface_id(), wall.surface_id)
        self.assertFalse(viewer.undo_window_button.isEnabled())

        viewer.cancel_window_placement(status_message=None)
        self.assertTrue(viewer.undo_window_button.isEnabled())
        viewer.select_wall_target(None)
        self.assertTrue(viewer.undo_window_button.isEnabled())
        viewer.undo_window_button.click()

        self.assertEqual(emitted, [True])
        self.assertFalse(viewer.is_window_placement_active())

    def test_hidden_semantic_wall_pick_selects_the_nearest_wall(self) -> None:
        viewer = self._build_viewer()
        wall = _build_wall()
        viewer.set_wall_targets((wall,))

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(2.0, 1.0),
        ):
            viewer._handle_window_wall_pick_requested(QPointF(20.0, 20.0))

        self.assertEqual(viewer.get_selected_wall_surface_id(), wall.surface_id)
        assert viewer.add_window_button is not None
        self.assertTrue(viewer.add_window_button.isEnabled())

    def test_valid_drag_emits_one_bounded_immutable_placement(self) -> None:
        viewer = self._build_viewer()
        wall = self._arm_selected_wall(viewer)
        emitted: list[object] = []
        viewer.window_placement_requested.connect(emitted.append)
        self._build_valid_preview(viewer)

        self.assertTrue(viewer._window_preview_is_valid)
        self.assertIsNotNone(viewer._window_preview_item)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(3.0, 2.0),
        ):
            viewer._handle_window_pointer_released(QPointF(40.0, 40.0))

        self.assertEqual(len(emitted), 1)
        placement = emitted[0]
        self.assertIsInstance(placement, WallWindowPlacement)
        assert isinstance(placement, WallWindowPlacement)
        self.assertEqual(placement.wall_surface_id, wall.surface_id)
        self.assertEqual(
            (
                placement.start_ratio,
                placement.end_ratio,
                placement.bottom_ratio,
                placement.top_ratio,
            ),
            (0.25, 0.75, 1.0 / 6.0, 2.0 / 3.0),
        )
        with self.assertRaises(FrozenInstanceError):
            placement.start_ratio = 0.0  # type: ignore[misc]
        self.assertFalse(viewer.is_window_placement_active())
        self.assertIsNone(viewer._window_preview_item)

    def test_out_of_wall_drag_shows_invalid_preview_and_emits_nothing(self) -> None:
        viewer = self._build_viewer()
        self._arm_selected_wall(viewer)
        emitted: list[object] = []
        viewer.window_placement_requested.connect(emitted.append)
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(1.0, 0.5),
        ):
            viewer._handle_window_pointer_pressed(QPointF())
        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(5.0, 2.0),
        ):
            viewer._handle_window_pointer_moved(QPointF(50.0, 50.0))

        self.assertFalse(viewer._window_preview_is_valid)
        self.assertIsNone(viewer._window_preview_placement)
        self.assertIsNotNone(viewer._window_preview_item)
        assert viewer.window_tools_status_label is not None
        self.assertNotEqual(
            viewer.window_tools_status_label.text(),
            "Release to add this window.",
        )

        with patch.object(
            viewer.view,
            "build_camera_ray",
            return_value=_ray_at(5.0, 2.0),
        ):
            viewer._handle_window_pointer_released(QPointF(50.0, 50.0))

        self.assertEqual(emitted, [])
        self.assertFalse(viewer.is_window_placement_active())

    def test_missing_release_ray_cannot_emit_a_stale_valid_preview(self) -> None:
        viewer = self._build_viewer()
        self._arm_selected_wall(viewer)
        emitted: list[object] = []
        viewer.window_placement_requested.connect(emitted.append)
        self._build_valid_preview(viewer)
        self.assertTrue(viewer._window_preview_is_valid)

        with patch.object(viewer.view, "build_camera_ray", return_value=None):
            viewer._handle_window_pointer_released(QPointF(80.0, 80.0))

        self.assertEqual(emitted, [])
        self.assertFalse(viewer._window_preview_is_valid)
        self.assertIsNone(viewer._window_preview_placement)
        self.assertIsNone(viewer._window_preview_item)

    def test_escape_cancels_preview_without_emitting(self) -> None:
        viewer = self._build_viewer()
        self._arm_selected_wall(viewer)
        emitted: list[object] = []
        viewer.window_placement_requested.connect(emitted.append)
        self._build_valid_preview(viewer)

        viewer.view.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        self.assertEqual(emitted, [])
        self.assertFalse(viewer.is_window_placement_active())
        self.assertIsNone(viewer._window_preview_item)
        assert viewer.add_window_button is not None
        self.assertFalse(viewer.add_window_button.isChecked())

    def test_selection_survives_scene_repopulation_and_stable_target_refresh(
        self,
    ) -> None:
        viewer = self._build_viewer()
        wall = _build_wall()
        viewer.set_wall_targets((wall,))
        viewer.select_wall_target(wall.surface_id)
        viewer.set_model(_build_model())
        first_outline = viewer._window_selection_item

        self.assertIsNotNone(first_outline)
        self.assertIn(first_outline, viewer.view.items)
        viewer.set_model(_build_model(), preserve_camera=True)

        self.assertEqual(viewer.get_selected_wall_surface_id(), wall.surface_id)
        self.assertIsNot(viewer._window_selection_item, first_outline)
        self.assertIn(viewer._window_selection_item, viewer.view.items)

        moved_wall = _build_wall(x_offset=1.0)
        viewer.set_wall_targets((moved_wall,))
        self.assertEqual(
            viewer.get_selected_wall_surface_id(),
            moved_wall.surface_id,
        )
        viewer.set_wall_targets(())
        self.assertIsNone(viewer.get_selected_wall_surface_id())
        self.assertIsNone(viewer._window_selection_item)


if __name__ == "__main__":
    unittest.main()
