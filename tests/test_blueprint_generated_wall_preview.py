# ### Environment setup ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import (
    GENERATED_WALL_PREVIEW_VERTEX_COLOR,
    BlueprintCanvas,
)
from housemaker.models import VertexData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_vertex_data(
    *,
    include_candidate: bool,
) -> VertexData:
    vertex_data = VertexData()
    first = vertex_data.add_vertex(10.0, 20.0)
    second = vertex_data.add_vertex(50.0, 20.0)
    vertex_data.add_edge(first.id, second.id)
    if include_candidate:
        third = vertex_data.add_vertex(90.0, 20.0)
        vertex_data.add_edge(second.id, third.id)
    return vertex_data


def _render_canvas(canvas: BlueprintCanvas) -> QImage:
    rendered = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    rendered.fill(Qt.GlobalColor.transparent)
    canvas.render(rendered)
    return rendered


# ### Tests ###
class BlueprintGeneratedWallPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = BlueprintCanvas()
        self.canvas.resize(640, 520)
        self.canvas.blueprint_image = QImage(
            100,
            100,
            QImage.Format.Format_RGB32,
        )
        self.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        self.canvas.vertex_data.copy_from(
            _build_vertex_data(include_candidate=False)
        )
        self.canvas.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()

    def test_setting_preview_snapshots_without_editing_canvas_state(self) -> None:
        candidate = _build_vertex_data(include_candidate=True)
        original_identity = id(self.canvas.vertex_data)
        original_payload = self.canvas.vertex_data.to_dict()
        original_undo_count = len(self.canvas.undo_stack)

        changed = self.canvas.set_generated_wall_preview(
            candidate,
            existing_edge_keys=((2, 1),),
        )
        candidate.move_vertex(3, 75.0, 75.0)

        self.assertTrue(changed)
        self.assertEqual(id(self.canvas.vertex_data), original_identity)
        self.assertEqual(self.canvas.vertex_data.to_dict(), original_payload)
        self.assertEqual(len(self.canvas.undo_stack), original_undo_count)
        preview = self.canvas.get_generated_wall_preview()
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(preview.existing_edge_keys, frozenset({(1, 2)}))
        self.assertEqual(
            (preview.vertices[-1].x, preview.vertices[-1].y),
            (90.0, 20.0),
        )

    def test_preview_uses_image_coordinates_and_omits_existing_edges(self) -> None:
        candidate = _build_vertex_data(include_candidate=True)
        anchored_midpoint = self.canvas._image_to_widget(30.0, 20.0).toPoint()
        without_preview = _render_canvas(self.canvas)
        self.canvas.set_generated_wall_preview(
            candidate,
            existing_edge_keys=((1, 2),),
        )

        rendered = _render_canvas(self.canvas)
        candidate_endpoint = self.canvas._image_to_widget(90.0, 20.0).toPoint()

        self.assertEqual(
            rendered.pixelColor(anchored_midpoint),
            without_preview.pixelColor(anchored_midpoint),
        )
        self.assertEqual(
            rendered.pixelColor(candidate_endpoint).name(),
            GENERATED_WALL_PREVIEW_VERTEX_COLOR.name(),
        )

    def test_clear_and_level_reset_remove_preview_without_geometry_edits(self) -> None:
        candidate = _build_vertex_data(include_candidate=True)
        self.canvas.set_generated_wall_preview(candidate)

        self.assertTrue(self.canvas.clear_generated_wall_preview())
        self.assertFalse(self.canvas.clear_generated_wall_preview())
        self.canvas.set_generated_wall_preview(candidate)
        self.canvas.set_level_data(
            vertex_data=self.canvas.vertex_data,
            rooms=[],
            image_path=None,
        )

        self.assertIsNone(self.canvas.get_generated_wall_preview())

    def test_commit_is_one_undoable_edit_and_preserves_graph_identity(self) -> None:
        candidate = _build_vertex_data(include_candidate=True)
        old_payload = self.canvas.vertex_data.to_dict()
        graph_identity = id(self.canvas.vertex_data)
        self.canvas.active_vertex_id = 1
        self.canvas.set_selected_vertex_ids((1,))
        self.canvas.preview_point = (25.0, 25.0)
        self.canvas.set_generated_wall_preview(candidate)
        geometry_events: list[str] = []
        self.canvas.geometry_changed.connect(
            lambda: geometry_events.append("geometry_changed")
        )

        committed = self.canvas.commit_generated_wall_preview()

        self.assertTrue(committed)
        self.assertEqual(id(self.canvas.vertex_data), graph_identity)
        self.assertEqual(self.canvas.vertex_data.to_dict(), candidate.to_dict())
        self.assertEqual(len(self.canvas.undo_stack), 1)
        self.assertEqual(geometry_events, ["geometry_changed"])
        self.assertIsNone(self.canvas.active_vertex_id)
        self.assertEqual(self.canvas.selected_vertex_ids, ())
        self.assertIsNone(self.canvas.preview_point)
        self.assertIsNone(self.canvas.get_generated_wall_preview())

        self.canvas.undo_last_step()

        self.assertEqual(id(self.canvas.vertex_data), graph_identity)
        self.assertEqual(self.canvas.vertex_data.to_dict(), old_payload)

    def test_commit_without_preview_is_a_no_op(self) -> None:
        original_payload = self.canvas.vertex_data.to_dict()
        geometry_events: list[str] = []
        self.canvas.geometry_changed.connect(
            lambda: geometry_events.append("geometry_changed")
        )

        self.assertFalse(self.canvas.commit_generated_wall_preview())
        self.assertEqual(self.canvas.vertex_data.to_dict(), original_payload)
        self.assertEqual(self.canvas.undo_stack, [])
        self.assertEqual(geometry_events, [])


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
