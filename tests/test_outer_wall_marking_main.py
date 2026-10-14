# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGroupBox, QPushButton

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.level_coordinates import (
    level_image_to_world_xy,
)
from housemaker.main import BlueprintWorkspace
from housemaker.models import LevelData, RoomData, VertexData
from housemaker.outer_wall_marking import project_world_points_to_level_image

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_outer_loop_level(
    level_index: int,
    *,
    scale: float = 1.0,
    offset_x_meters: float = 0.0,
    offset_y_meters: float = 0.0,
) -> tuple[LevelData, tuple[int, ...]]:
    """Return one level with a square exterior and one interior wall."""

    vertex_data = VertexData()
    outer_ids = tuple(
        vertex_data.add_vertex(*point).id
        for point in (
            (40.0, 40.0),
            (200.0, 40.0),
            (200.0, 200.0),
            (40.0, 200.0),
        )
    )
    for start_id, end_id in zip(
        outer_ids,
        (*outer_ids[1:], outer_ids[0]),
    ):
        vertex_data.add_edge(start_id, end_id)

    interior_start = vertex_data.add_vertex(90.0, 100.0)
    interior_end = vertex_data.add_vertex(150.0, 100.0)
    vertex_data.add_edge(interior_start.id, interior_end.id)
    room_center = vertex_data.add_vertex(120.0, 120.0)
    room = RoomData(
        name=f"Room {level_index}",
        vertex_ids=outer_ids,
        center_vertex_id=room_center.id,
        color_rgb=(120, 150, 180),
    )
    return (
        LevelData(
            index=level_index,
            name=f"Level {level_index}",
            scale=scale,
            offset_x_meters=offset_x_meters,
            offset_y_meters=offset_y_meters,
            image_size_pixels=(240.0, 240.0),
            vertex_data=vertex_data,
            rooms=[room],
        ),
        outer_ids,
    )


def _build_empty_level(
    level_index: int,
    *,
    scale: float = 1.0,
    offset_x_meters: float = 0.0,
    offset_y_meters: float = 0.0,
) -> LevelData:
    """Return an empty target level with a known plan coordinate system."""

    return LevelData(
        index=level_index,
        name=f"Level {level_index}",
        scale=scale,
        offset_x_meters=offset_x_meters,
        offset_y_meters=offset_y_meters,
        image_size_pixels=(240.0, 240.0),
        vertex_data=VertexData(),
    )


def _mark_keys(workspace: BlueprintWorkspace) -> set[tuple[int, int]]:
    """Return stable level-and-vertex identities from the mark model."""

    return {
        (mark.level_index, mark.vertex_id)
        for mark in workspace.outer_wall_marks
    }


def _topology_snapshot(
    workspace: BlueprintWorkspace,
) -> tuple[tuple[int, dict[str, object]], ...]:
    """Return all editable wall topology without any marking metadata."""

    return tuple(
        (level.index, level.vertex_data.to_dict())
        for level in workspace.levels
    )


# ### Outer wall marking integration tests ###
class OuterWallMarkingMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                Path(self.temporary_directory.name) / "settings.json"
            )
        )
        self.source_level, self.outer_ids = _build_outer_loop_level(
            2,
            scale=1.25,
            offset_x_meters=0.4,
            offset_y_meters=-0.3,
        )
        self.target_level = _build_empty_level(
            3,
            scale=0.8,
            offset_x_meters=-0.6,
            offset_y_meters=0.2,
        )
        self.workspace._apply_project_state(
            [self.source_level, self.target_level],
            0,
        )
        self.workspace.resize(1500, 900)
        self.workspace.show()
        self.workspace.canvas.blueprint_image = QImage(
            240,
            240,
            QImage.Format.Format_RGB32,
        )
        self.workspace.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        _qt_application.processEvents()

        groups = [
            group
            for group in self.workspace.side_panel.findChildren(QGroupBox)
            if group.title() == "Outer wall marking"
        ]
        self.assertEqual(len(groups), 1)
        self.marking_group = groups[0]

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        self.temporary_directory.cleanup()
        _qt_application.processEvents()

    def _button(self, text: str) -> QPushButton:
        matches = [
            button
            for button in self.marking_group.findChildren(QPushButton)
            if button.text() == text
        ]
        self.assertEqual(len(matches), 1, f"Expected one {text!r} button")
        return matches[0]

    def _click_local_vertex(self, vertex_id: int) -> None:
        vertex = self.source_level.vertex_data.get_vertex(vertex_id)
        assert vertex is not None
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=self.workspace.canvas._image_to_widget(
                vertex.x,
                vertex.y,
            ).toPoint(),
        )
        _qt_application.processEvents()

    def test_manual_mark_changes_only_marking_metadata(self) -> None:
        before = _topology_snapshot(self.workspace)
        selected_ids = self.outer_ids[:2]
        self.workspace.canvas.set_selected_vertex_ids(selected_ids)

        self._button("Mark").click()

        self.assertEqual(
            _mark_keys(self.workspace),
            {(self.source_level.index, vertex_id) for vertex_id in selected_ids},
        )
        self.assertEqual(_topology_snapshot(self.workspace), before)
        self.assertEqual(len(self.target_level.vertex_data.vertices), 0)
        self.assertTrue(self._button("Edit").isEnabled())

    def test_global_undo_restores_marks_without_touching_wall_topology(self) -> None:
        before = _topology_snapshot(self.workspace)
        selected_ids = self.outer_ids[:2]
        self.workspace.canvas.set_selected_vertex_ids(selected_ids)
        self._button("Mark").click()

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(_mark_keys(self.workspace), set())
        self.assertEqual(_topology_snapshot(self.workspace), before)
        self.assertEqual(
            self.workspace.canvas.selected_vertex_ids,
            selected_ids,
        )

    def test_edit_previews_existing_marks_and_confirms_one_change(self) -> None:
        before = _topology_snapshot(self.workspace)
        initially_marked_ids = self.outer_ids[:2]
        removed_vertex_id = initially_marked_ids[0]
        added_vertex_id = self.source_level.vertex_data.vertices[4].id
        self.workspace.canvas.set_selected_vertex_ids(initially_marked_ids)
        self._button("Mark").click()
        baseline_keys = _mark_keys(self.workspace)
        edit_button = self._button("Edit")
        auto_mark_button = self._button("Auto mark")

        edit_button.click()

        self.assertEqual(edit_button.text(), "Confirm")
        self.assertFalse(auto_mark_button.isEnabled())
        self.assertTrue(self.workspace.canvas.is_outer_wall_marking_active())
        self.assertEqual(_mark_keys(self.workspace), baseline_keys)

        self._click_local_vertex(removed_vertex_id)
        self._click_local_vertex(added_vertex_id)

        expected_ids = {initially_marked_ids[1], added_vertex_id}
        self.assertEqual(_mark_keys(self.workspace), baseline_keys)
        self.assertEqual(
            self.workspace.canvas.get_outer_wall_mark_vertex_ids(),
            expected_ids,
        )

        edit_button.click()

        self.assertEqual(edit_button.text(), "Edit")
        self.assertTrue(auto_mark_button.isEnabled())
        self.assertFalse(self.workspace.canvas.is_outer_wall_marking_active())
        self.assertEqual(
            _mark_keys(self.workspace),
            {
                (self.source_level.index, vertex_id)
                for vertex_id in expected_ids
            },
        )
        self.assertEqual(_topology_snapshot(self.workspace), before)

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(_mark_keys(self.workspace), baseline_keys)
        self.assertEqual(_topology_snapshot(self.workspace), before)

    def test_escape_cancels_edit_and_restores_committed_marks(self) -> None:
        initially_marked_ids = self.outer_ids[:2]
        self.workspace.canvas.set_selected_vertex_ids(initially_marked_ids)
        self._button("Mark").click()
        baseline_keys = _mark_keys(self.workspace)
        undo_count = len(self.workspace._canvas_undo_stack)
        edit_button = self._button("Edit")
        edit_button.click()
        self._click_local_vertex(initially_marked_ids[0])

        QTest.keyClick(self.workspace.canvas, Qt.Key.Key_Escape)
        _qt_application.processEvents()

        self.assertEqual(edit_button.text(), "Edit")
        self.assertFalse(self.workspace.canvas.is_outer_wall_marking_active())
        self.assertEqual(_mark_keys(self.workspace), baseline_keys)
        self.assertEqual(
            self.workspace.canvas.get_outer_wall_mark_vertex_ids(),
            set(initially_marked_ids),
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)

    def test_auto_mark_previews_outer_loop_and_confirm_commits_it(self) -> None:
        before = _topology_snapshot(self.workspace)
        auto_mark_button = self._button("Auto mark")

        auto_mark_button.click()

        self.assertEqual(auto_mark_button.text(), "Confirm")
        self.assertFalse(self._button("Edit").isEnabled())
        self.assertEqual(_mark_keys(self.workspace), set())

        auto_mark_button.click()

        self.assertEqual(auto_mark_button.text(), "Auto mark")
        self.assertTrue(self._button("Edit").isEnabled())
        self.assertEqual(
            _mark_keys(self.workspace),
            {
                (self.source_level.index, vertex_id)
                for vertex_id in self.outer_ids
            },
        )
        self.assertEqual(_topology_snapshot(self.workspace), before)
        self.assertEqual(len(self.target_level.vertex_data.vertices), 0)

    def test_auto_mark_allows_manual_preview_toggles_before_confirm(self) -> None:
        auto_mark_button = self._button("Auto mark")
        removed_vertex_id = self.outer_ids[1]
        added_interior_vertex_id = self.source_level.vertex_data.vertices[4].id
        auto_mark_button.click()

        self._click_local_vertex(removed_vertex_id)
        self._click_local_vertex(added_interior_vertex_id)
        auto_mark_button.click()

        expected_ids = set(self.outer_ids)
        expected_ids.remove(removed_vertex_id)
        expected_ids.add(added_interior_vertex_id)
        self.assertEqual(
            _mark_keys(self.workspace),
            {
                (self.source_level.index, vertex_id)
                for vertex_id in expected_ids
            },
        )

    def test_right_click_cancels_auto_mark_without_committing(self) -> None:
        before = _topology_snapshot(self.workspace)
        auto_mark_button = self._button("Auto mark")
        auto_mark_button.click()
        self._click_local_vertex(self.outer_ids[0])

        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.RightButton,
            pos=self.workspace.canvas.rect().center(),
        )
        _qt_application.processEvents()

        self.assertEqual(auto_mark_button.text(), "Auto mark")
        self.assertEqual(_mark_keys(self.workspace), set())
        self.assertEqual(_topology_snapshot(self.workspace), before)

    def test_other_level_guides_snap_without_materializing_geometry(self) -> None:
        source_vertex_ids = self.outer_ids[:2]
        source_vertices = tuple(
            self.source_level.vertex_data.get_vertex(vertex_id)
            for vertex_id in source_vertex_ids
        )
        assert all(vertex is not None for vertex in source_vertices)
        self.workspace.canvas.set_selected_vertex_ids(source_vertex_ids)
        self._button("Mark").click()
        topology_after_mark = _topology_snapshot(self.workspace)

        self.workspace._handle_level_selection_changed(1)
        self.workspace.canvas.blueprint_image = QImage(
            240,
            240,
            QImage.Format.Format_RGB32,
        )
        self.workspace.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        _qt_application.processEvents()
        guide_points = project_world_points_to_level_image(
            self.target_level,
            tuple(
                level_image_to_world_xy(
                    self.source_level,
                    source_vertex.x,
                    source_vertex.y,
                )
                for source_vertex in source_vertices
                if source_vertex is not None
            ),
        )
        overlay = self.workspace.canvas.get_outer_wall_reference_overlay()

        self.assertEqual(set(overlay.points), set(guide_points))
        self.assertEqual(
            {frozenset(edge) for edge in overlay.edges},
            {frozenset(guide_points)},
        )

        preview = self.workspace.canvas._build_connection_preview(
            QPointF(guide_points[0][0] + 1.0, guide_points[0][1] + 1.0),
            Qt.KeyboardModifier.NoModifier,
        )

        self.assertAlmostEqual(preview.point[0], guide_points[0][0])
        self.assertAlmostEqual(preview.point[1], guide_points[0][1])
        self.assertEqual(_topology_snapshot(self.workspace), topology_after_mark)
        self.assertEqual(len(self.target_level.vertex_data.vertices), 0)

    def test_auto_mark_availability_tracks_wall_topology_changes(self) -> None:
        self.workspace._handle_level_selection_changed(1)
        auto_mark_button = self._button("Auto mark")
        self.assertFalse(auto_mark_button.isEnabled())

        first = self.target_level.vertex_data.add_vertex(20.0, 20.0)
        second = self.target_level.vertex_data.add_vertex(80.0, 20.0)
        self.target_level.vertex_data.add_edge(first.id, second.id)
        self.workspace._handle_canvas_surface_geometry_changed()

        self.assertTrue(auto_mark_button.isEnabled())

        self.target_level.vertex_data.delete_vertex(first.id)
        self.target_level.vertex_data.delete_vertex(second.id)
        self.workspace._handle_canvas_surface_geometry_changed()

        self.assertFalse(auto_mark_button.isEnabled())


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
