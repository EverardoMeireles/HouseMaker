# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from dataclasses import dataclass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import (
    WALL_VERIFICATION_EXCLUDED_COLOR,
    WALL_VERIFICATION_VERTEX_COLOR,
    BlueprintCanvas,
)
from housemaker.models import Vertex

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
@dataclass(frozen=True)
class _WallAdditionProposal:
    """Match the backend value contract without importing its network code."""

    proposal_id: str
    start: Vertex
    end: Vertex
    confidence: float = 0.95
    reason: str = "The plan shows a continuous wall."


def _build_proposals() -> tuple[_WallAdditionProposal, ...]:
    return (
        _WallAdditionProposal(
            "first", Vertex(2, 50.0, 20.0), Vertex(3, 90.0, 20.0)
        ),
        _WallAdditionProposal(
            "second", Vertex(3, 90.0, 20.0), Vertex(4, 90.0, 65.0)
        ),
    )


def _render_canvas(canvas: BlueprintCanvas) -> QImage:
    rendered = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    rendered.fill(Qt.GlobalColor.transparent)
    canvas.render(rendered)
    return rendered


# ### Tests ###
class BlueprintWallVerificationPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = BlueprintCanvas()
        self.canvas.resize(640, 520)
        self.canvas.blueprint_image = QImage(
            100, 100, QImage.Format.Format_RGB32
        )
        self.canvas.blueprint_image.fill(Qt.GlobalColor.white)
        first = self.canvas.vertex_data.add_vertex(10.0, 20.0)
        second = self.canvas.vertex_data.add_vertex(50.0, 20.0)
        self.canvas.vertex_data.add_edge(first.id, second.id)
        self.canvas.show()
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.canvas.close()
        self.canvas.deleteLater()
        _qt_application.processEvents()

    def _click_image_point(self, image_x: float, image_y: float) -> None:
        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=self.canvas._image_to_widget(image_x, image_y).toPoint(),
        )

    def test_preview_selects_all_without_geometry_or_undo_changes(self) -> None:
        original_payload = self.canvas.vertex_data.to_dict()
        original_identity = id(self.canvas.vertex_data)
        geometry_events: list[bool] = []
        self.canvas.geometry_changed.connect(lambda: geometry_events.append(True))

        self.canvas.set_wall_verification_preview(_build_proposals())

        self.assertEqual(
            self.canvas.selected_wall_verification_proposal_ids(),
            frozenset({"first", "second"}),
        )
        self.assertEqual(self.canvas.vertex_data.to_dict(), original_payload)
        self.assertEqual(id(self.canvas.vertex_data), original_identity)
        self.assertEqual(self.canvas.undo_stack, [])
        self.assertEqual(geometry_events, [])
        self.assertIsNone(self.canvas.get_generated_wall_preview())

    def test_edge_click_toggles_only_that_proposal(self) -> None:
        self.canvas.set_wall_verification_preview(_build_proposals())
        selection_events: list[bool] = []
        self.canvas.wall_verification_selection_changed.connect(
            lambda: selection_events.append(True)
        )

        self._click_image_point(70.0, 20.0)
        self.assertEqual(
            self.canvas.selected_wall_verification_proposal_ids(),
            frozenset({"second"}),
        )
        self._click_image_point(70.0, 20.0)
        self.assertEqual(
            self.canvas.selected_wall_verification_proposal_ids(),
            frozenset({"first", "second"}),
        )
        self.assertEqual(len(selection_events), 2)

    def test_shared_vertex_toggles_all_incident_proposals_together(self) -> None:
        self.canvas.set_wall_verification_preview(_build_proposals())

        self._click_image_point(90.0, 20.0)
        self.assertEqual(
            self.canvas.selected_wall_verification_proposal_ids(), frozenset()
        )
        self._click_image_point(90.0, 20.0)
        self.assertEqual(
            self.canvas.selected_wall_verification_proposal_ids(),
            frozenset({"first", "second"}),
        )

    def test_blank_click_does_not_create_vertex_or_start_existing_edit_tool(self) -> None:
        self.canvas.set_wall_verification_preview(_build_proposals())
        original_payload = self.canvas.vertex_data.to_dict()
        self.canvas._plan_image_erase_active = True

        self._click_image_point(25.0, 60.0)

        self.assertEqual(self.canvas.vertex_data.to_dict(), original_payload)
        self.assertIsNone(self.canvas._plan_image_erase_selection_start)
        self.assertEqual(self.canvas.undo_stack, [])

    def test_deletion_key_does_not_delete_previous_wall_selection(self) -> None:
        self.canvas.set_selected_vertex_ids((1, 2))
        self.canvas.set_wall_verification_preview(_build_proposals())
        original_payload = self.canvas.vertex_data.to_dict()

        QTest.keyClick(self.canvas, Qt.Key.Key_Delete)

        self.assertEqual(self.canvas.vertex_data.to_dict(), original_payload)
        self.assertEqual(self.canvas.selected_vertex_ids, (1, 2))

    def test_hit_radius_stays_in_screen_pixels_when_zoomed(self) -> None:
        self.canvas.set_wall_verification_preview(_build_proposals())
        self.canvas.zoom_scale *= 2.0
        edge_position = self.canvas._image_to_widget(70.0, 20.0).toPoint()

        QTest.mouseClick(
            self.canvas,
            Qt.MouseButton.LeftButton,
            pos=edge_position + QPoint(0, 7),
        )

        self.assertEqual(
            self.canvas.selected_wall_verification_proposal_ids(),
            frozenset({"second"}),
        )

    def test_right_click_dismisses_review_before_other_active_modes(self) -> None:
        self.canvas.set_wall_verification_preview(_build_proposals())
        self.canvas._outer_wall_marking_active = True
        dismissed: list[bool] = []
        cancelled_markings: list[bool] = []
        self.canvas.wall_verification_dismiss_requested.connect(
            lambda: dismissed.append(True)
        )
        self.canvas.outer_wall_mark_cancel_requested.connect(
            lambda: cancelled_markings.append(True)
        )

        QTest.mouseClick(
            self.canvas, Qt.MouseButton.RightButton, pos=QPoint(20, 20)
        )

        self.assertEqual(dismissed, [True])
        self.assertEqual(cancelled_markings, [])

    def test_pending_review_can_be_dismissed_without_an_image(self) -> None:
        self.canvas.blueprint_image = None
        self.canvas.set_wall_verification_pending(True)
        dismissed: list[bool] = []
        self.canvas.wall_verification_dismiss_requested.connect(
            lambda: dismissed.append(True)
        )

        QTest.mouseClick(self.canvas, Qt.MouseButton.RightButton)
        QTest.keyClick(self.canvas, Qt.Key.Key_Escape)

        self.assertEqual(dismissed, [True, True])

    def test_clear_and_level_replacement_remove_overlay_and_pending_state(self) -> None:
        self.canvas.set_wall_verification_preview(_build_proposals())
        original_payload = self.canvas.vertex_data.to_dict()

        self.assertTrue(self.canvas.clear_wall_verification_preview())
        self.assertFalse(self.canvas.clear_wall_verification_preview())
        self.assertEqual(self.canvas.vertex_data.to_dict(), original_payload)
        self.canvas.set_wall_verification_preview(_build_proposals())
        self.canvas.set_wall_verification_pending(True)
        self.canvas.set_level_data(
            self.canvas.vertex_data, rooms=[], image_path=None
        )

        self.assertEqual(self.canvas._wall_verification_preview, ())
        self.assertFalse(self.canvas._wall_verification_pending)
        self.assertEqual(
            self.canvas.selected_wall_verification_proposal_ids(), frozenset()
        )
        self.assertFalse(self.canvas.clear_wall_verification_preview())

    def test_overlay_is_red_and_excluded_proposals_remain_visible(self) -> None:
        self.canvas.set_wall_verification_preview(_build_proposals())
        endpoint = self.canvas._image_to_widget(90.0, 65.0).toPoint()

        rendered = _render_canvas(self.canvas)
        self.assertEqual(
            rendered.pixelColor(endpoint).name(),
            WALL_VERIFICATION_VERTEX_COLOR.name(),
        )
        self._click_image_point(90.0, 45.0)
        rendered = _render_canvas(self.canvas)
        self.assertEqual(
            rendered.pixelColor(endpoint).name(),
            WALL_VERIFICATION_EXCLUDED_COLOR.name(),
        )

    def test_duplicate_proposal_ids_are_rejected(self) -> None:
        first, second = _build_proposals()
        duplicate = _WallAdditionProposal("first", second.start, second.end)

        with self.assertRaisesRegex(ValueError, "unique IDs"):
            self.canvas.set_wall_verification_preview((first, duplicate))

    def test_nonfinite_proposal_coordinates_are_rejected(self) -> None:
        invalid = _WallAdditionProposal(
            "invalid", Vertex(3, float("nan"), 20.0), Vertex(4, 50.0, 60.0)
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            self.canvas.set_wall_verification_preview((invalid,))


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
