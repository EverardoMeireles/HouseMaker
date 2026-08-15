# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.blueprint_canvas import (
    BlueprintCanvas,
    STAIR_FLOATING_WITH_RISER_COLOR,
    StairPlacement,
    StairSectionPlacement,
    _get_stair_style_color,
    _is_floating_stair_style,
    _normalize_stair_style,
)
from housemaker.models import (
    STAIR_STYLE_FLOATING,
    STAIR_STYLE_FLOATING_WITH_RISER,
    STAIR_STYLE_SUPPORTED,
    LevelData,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_level(level_index: int) -> LevelData:
    return LevelData(
        index=level_index,
        name=f"Level {level_index}",
        image_size_pixels=(100.0, 100.0),
    )


def _set_canvas_level(
    canvas: BlueprintCanvas,
    level: LevelData,
    stairs: list[object] | None = None,
) -> None:
    canvas.set_level_data(
        vertex_data=level.vertex_data,
        rooms=level.rooms,
        image_path=None,
        image_scale=level.image_scale,
        image_offset_x=level.image_offset_x,
        image_offset_y=level.image_offset_y,
        floor_contour_vertex_ids=level.floor_contour_vertex_ids,
        doorways=level.doorways,
    )
    canvas.blueprint_image = QImage(100, 100, QImage.Format.Format_RGB32)
    canvas.blueprint_image.fill(Qt.GlobalColor.white)
    canvas.set_stair_context(stairs or [], level)


def _image_position(canvas: BlueprintCanvas, x: float, y: float):
    return canvas._image_to_widget(x, y).toPoint()


def _place_point(canvas: BlueprintCanvas, x: float, y: float) -> None:
    canvas._place_stair_endpoint(QPointF(x, y), Qt.NoModifier)


def _four_point_stair_dict() -> dict[str, object]:
    return {
        "style": STAIR_STYLE_SUPPORTED,
        "start_level_index": 2,
        "start_a_x": 20.0,
        "start_a_y": 30.0,
        "start_b_x": 50.0,
        "start_b_y": 30.0,
        "end_level_index": 3,
        "end_a_x": 24.0,
        "end_a_y": 70.0,
        "end_b_x": 54.0,
        "end_b_y": 70.0,
    }


# ### Canvas stair interaction tests ###
class StairCanvasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widgets: list[BlueprintCanvas] = []

    def tearDown(self) -> None:
        for widget in reversed(self.widgets):
            widget.close()
        _qt_application.processEvents()

    def _build_canvas(self, level: LevelData) -> BlueprintCanvas:
        canvas = BlueprintCanvas()
        canvas.resize(640, 520)
        _set_canvas_level(canvas, level)
        canvas.show()
        _qt_application.processEvents()
        self.widgets.append(canvas)
        return canvas

    def test_four_point_placement_accepts_free_points_across_two_levels(
        self,
    ) -> None:
        start_level = _build_level(2)
        end_level = _build_level(3)
        canvas = self._build_canvas(start_level)
        starts: list[object] = []
        ready: list[object] = []
        completed: list[object] = []
        canvas.stair_start_placed.connect(starts.append)
        canvas.stair_placement_ready.connect(ready.append)
        canvas.stair_placement_completed.connect(completed.append)

        canvas.start_stair_placement(STAIR_STYLE_FLOATING)
        _place_point(canvas, 20.0, 20.0)
        self.assertIsNone(canvas.get_pending_stair_placement())
        self.assertIsNotNone(canvas.pending_stair_point)

        _place_point(canvas, 50.0, 20.0)
        pending = canvas.get_pending_stair_placement()
        self.assertIsNotNone(pending)
        self.assertEqual(starts, [pending])
        self.assertEqual(
            (
                pending.start_a_x,  # type: ignore[union-attr]
                pending.start_a_y,  # type: ignore[union-attr]
                pending.start_b_x,  # type: ignore[union-attr]
                pending.start_b_y,  # type: ignore[union-attr]
            ),
            (20.0, 20.0, 50.0, 20.0),
        )

        _set_canvas_level(canvas, end_level)
        _place_point(canvas, 25.0, 70.0)
        self.assertEqual(canvas.get_pending_stair_placement(), pending)
        self.assertIsNotNone(canvas.pending_stair_point)
        self.assertEqual(completed, [])

        _place_point(canvas, 55.0, 70.0)
        self.assertEqual(completed, [])
        self.assertEqual(len(ready), 1)
        self.assertTrue(canvas.is_stair_placement_active())
        self.assertTrue(canvas.is_stair_ready_for_confirmation())
        self.assertTrue(canvas.confirm_stair_placement())
        self.assertEqual(len(completed), 1)
        placement = completed[0]
        self.assertIsInstance(placement, StairPlacement)
        self.assertEqual(placement.style, STAIR_STYLE_FLOATING)
        self.assertEqual(placement.start_level_index, 2)
        self.assertEqual(placement.end_level_index, 3)
        self.assertEqual(
            (
                placement.start_a_x,
                placement.start_a_y,
                placement.start_b_x,
                placement.start_b_y,
            ),
            (20.0, 20.0, 50.0, 20.0),
        )
        self.assertEqual(
            (
                placement.end_a_x,
                placement.end_a_y,
                placement.end_b_x,
                placement.end_b_y,
            ),
            (25.0, 70.0, 55.0, 70.0),
        )
        self.assertIsNone(canvas.get_pending_stair_placement())
        self.assertIsNone(canvas.pending_stair_point)
        self.assertEqual(start_level.vertex_data.vertices, [])
        self.assertEqual(end_level.vertex_data.vertices, [])

    def test_floating_with_riser_style_uses_the_floating_canvas_marker(
        self,
    ) -> None:
        level = _build_level(2)
        canvas = self._build_canvas(level)

        canvas.start_stair_placement(STAIR_STYLE_FLOATING_WITH_RISER)

        self.assertEqual(
            canvas.pending_stair_style,
            STAIR_STYLE_FLOATING_WITH_RISER,
        )
        self.assertEqual(
            _normalize_stair_style(STAIR_STYLE_FLOATING_WITH_RISER),
            STAIR_STYLE_FLOATING_WITH_RISER,
        )
        self.assertTrue(
            _is_floating_stair_style(STAIR_STYLE_FLOATING_WITH_RISER)
        )
        self.assertEqual(
            _get_stair_style_color(STAIR_STYLE_FLOATING_WITH_RISER),
            STAIR_FLOATING_WITH_RISER_COLOR,
        )

    def test_points_bind_to_vertices_and_snap_to_edges_without_mutation(
        self,
    ) -> None:
        start_level = _build_level(2)
        end_level = _build_level(3)
        start_a = start_level.vertex_data.add_vertex(20.0, 25.0)
        start_b = start_level.vertex_data.add_vertex(60.0, 25.0)
        wall_a = end_level.vertex_data.add_vertex(10.0, 70.0)
        wall_b = end_level.vertex_data.add_vertex(80.0, 70.0)
        end_level.vertex_data.add_edge(wall_a.id, wall_b.id)
        canvas = self._build_canvas(start_level)
        completed: list[object] = []
        canvas.stair_placement_completed.connect(completed.append)
        original_start_data = start_level.vertex_data.to_dict()
        original_end_data = end_level.vertex_data.to_dict()

        canvas.start_stair_placement()
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, start_a.x, start_a.y),
        )
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, start_b.x, start_b.y),
        )
        _set_canvas_level(canvas, end_level)
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, 35.0, 69.0),
        )
        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, wall_b.x, wall_b.y),
        )

        self.assertTrue(canvas.confirm_stair_placement())
        self.assertEqual(len(completed), 1)
        placement = completed[0]
        self.assertEqual(placement.start_a_vertex_id, start_a.id)
        self.assertEqual(placement.start_b_vertex_id, start_b.id)
        self.assertIsNone(placement.end_a_vertex_id)
        self.assertEqual(placement.end_b_vertex_id, wall_b.id)
        self.assertAlmostEqual(placement.end_a_x, 35.0, places=1)
        self.assertAlmostEqual(placement.end_a_y, 70.0, places=1)
        self.assertEqual(start_level.vertex_data.to_dict(), original_start_data)
        self.assertEqual(end_level.vertex_data.to_dict(), original_end_data)

    def test_refinement_pairs_are_ordered_and_keep_current_level_ownership(
        self,
    ) -> None:
        start_level = _build_level(2)
        end_level = _build_level(3)
        end_vertex = end_level.vertex_data.add_vertex(30.0, 52.0)
        canvas = self._build_canvas(start_level)
        ready: list[StairPlacement] = []
        completed: list[StairPlacement] = []
        canvas.stair_placement_ready.connect(ready.append)
        canvas.stair_placement_completed.connect(completed.append)

        canvas.start_stair_placement(STAIR_STYLE_SUPPORTED)
        _place_point(canvas, 10.0, 20.0)
        _place_point(canvas, 40.0, 20.0)
        _set_canvas_level(canvas, end_level)
        _place_point(canvas, 50.0, 80.0)
        _place_point(canvas, 80.0, 80.0)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, end_vertex.x, end_vertex.y),
        )
        _place_point(canvas, 60.0, 52.0)
        _set_canvas_level(canvas, start_level)
        _place_point(canvas, 20.0, 38.0)
        _place_point(canvas, 50.0, 38.0)

        draft = canvas.get_stair_placement_draft()
        self.assertIsNotNone(draft)
        self.assertEqual(len(ready), 3)
        self.assertEqual(completed, [])
        self.assertEqual(
            draft.intermediate_sections,  # type: ignore[union-attr]
            (
                StairSectionPlacement(
                    level_index=3,
                    a_x=30.0,
                    a_y=52.0,
                    b_x=60.0,
                    b_y=52.0,
                    a_vertex_id=end_vertex.id,
                ),
                StairSectionPlacement(
                    level_index=2,
                    a_x=20.0,
                    a_y=38.0,
                    b_x=50.0,
                    b_y=38.0,
                ),
            ),
        )

        self.assertTrue(canvas.confirm_stair_placement())
        self.assertEqual(completed, [draft])
        self.assertFalse(canvas.is_stair_placement_active())
        self.assertIsNone(canvas.get_stair_placement_draft())

    def test_restarting_placement_does_not_replace_an_active_curve_draft(
        self,
    ) -> None:
        start_level = _build_level(2)
        end_level = _build_level(3)
        canvas = self._build_canvas(start_level)

        canvas.start_stair_placement(STAIR_STYLE_SUPPORTED)
        _place_point(canvas, 10.0, 20.0)
        _place_point(canvas, 40.0, 20.0)
        _set_canvas_level(canvas, end_level)
        _place_point(canvas, 50.0, 80.0)
        _place_point(canvas, 80.0, 80.0)
        _set_canvas_level(canvas, start_level)
        _place_point(canvas, 20.0, 42.0)
        _place_point(canvas, 50.0, 42.0)
        initial_draft = canvas.get_stair_placement_draft()

        self.assertIsNotNone(initial_draft)
        canvas.start_stair_placement(STAIR_STYLE_FLOATING)

        self.assertEqual(canvas.get_stair_placement_draft(), initial_draft)
        self.assertEqual(canvas.pending_stair_style, STAIR_STYLE_SUPPORTED)
        _place_point(canvas, 25.0, 60.0)
        _place_point(canvas, 55.0, 60.0)

        updated_draft = canvas.get_stair_placement_draft()
        self.assertIsNotNone(updated_draft)
        self.assertEqual(
            updated_draft.intermediate_sections,  # type: ignore[union-attr]
            (
                StairSectionPlacement(2, 20.0, 42.0, 50.0, 42.0),
                StairSectionPlacement(2, 25.0, 60.0, 55.0, 60.0),
            ),
        )
        self.assertEqual(
            (
                updated_draft.start_a_x,  # type: ignore[union-attr]
                updated_draft.start_a_y,
                updated_draft.start_b_x,
                updated_draft.start_b_y,
                updated_draft.end_a_x,
                updated_draft.end_a_y,
                updated_draft.end_b_x,
                updated_draft.end_b_y,
            ),
            (10.0, 20.0, 40.0, 20.0, 50.0, 80.0, 80.0, 80.0),
        )

    def test_route_continuity_connects_consecutive_sections_by_level(
        self,
    ) -> None:
        start_level = _build_level(2)
        end_level = _build_level(3)
        canvas = self._build_canvas(start_level)
        placement = StairPlacement(
            style=STAIR_STYLE_SUPPORTED,
            start_level_index=2,
            start_a_x=10.0,
            start_a_y=20.0,
            start_b_x=40.0,
            start_b_y=20.0,
            end_level_index=3,
            end_a_x=60.0,
            end_a_y=80.0,
            end_b_x=90.0,
            end_b_y=80.0,
            intermediate_sections=(
                StairSectionPlacement(2, 20.0, 42.0, 50.0, 42.0),
                StairSectionPlacement(3, 45.0, 62.0, 75.0, 62.0),
            ),
        )

        with (
            patch.object(
                canvas,
                "_paint_stair_route_centerline",
            ) as paint_centerline,
            patch.object(
                canvas,
                "_paint_stair_route_level_transition",
            ) as paint_transition,
        ):
            canvas._paint_stair_route_continuity(
                object(),  # type: ignore[arg-type]
                placement,
                STAIR_STYLE_SUPPORTED,
                pending=False,
            )

        self.assertEqual(paint_centerline.call_count, 1)
        self.assertEqual(paint_transition.call_count, 1)
        self.assertEqual(
            paint_transition.call_args.kwargs["source_level_index"],
            2,
        )
        self.assertEqual(
            paint_transition.call_args.kwargs["destination_level_index"],
            3,
        )
        self.assertEqual(
            paint_transition.call_args.kwargs["label"],
            "Curve route to",
        )

        _set_canvas_level(canvas, end_level)
        with (
            patch.object(
                canvas,
                "_paint_stair_route_centerline",
            ) as paint_centerline,
            patch.object(
                canvas,
                "_paint_stair_route_level_transition",
            ) as paint_transition,
        ):
            canvas._paint_stair_route_continuity(
                object(),  # type: ignore[arg-type]
                placement,
                STAIR_STYLE_SUPPORTED,
                pending=False,
            )

        self.assertEqual(paint_centerline.call_count, 1)
        self.assertEqual(paint_transition.call_count, 1)
        self.assertEqual(
            paint_transition.call_args.kwargs["source_level_index"],
            3,
        )
        self.assertEqual(
            paint_transition.call_args.kwargs["destination_level_index"],
            2,
        )
        self.assertEqual(
            paint_transition.call_args.kwargs["label"],
            "Curve route from",
        )

    def test_confirm_rejects_half_pair_and_backspace_recovers(self) -> None:
        start_level = _build_level(2)
        end_level = _build_level(3)
        canvas = self._build_canvas(start_level)
        errors: list[str] = []
        canvas.stair_placement_invalid_endpoint.connect(errors.append)

        canvas.start_stair_placement()
        _place_point(canvas, 10.0, 20.0)
        _place_point(canvas, 40.0, 20.0)
        _set_canvas_level(canvas, end_level)
        _place_point(canvas, 50.0, 80.0)
        _place_point(canvas, 80.0, 80.0)
        _place_point(canvas, 30.0, 50.0)

        self.assertFalse(canvas.is_stair_ready_for_confirmation())
        self.assertFalse(canvas.confirm_stair_placement())
        self.assertEqual(
            errors,
            [
                "Place the second point of the current stair section before "
                "confirming."
            ],
        )

        QTest.keyClick(canvas, Qt.Key.Key_Backspace)
        self.assertIsNone(canvas.get_pending_stair_point())
        self.assertTrue(canvas.is_stair_ready_for_confirmation())

    def test_backspace_removes_only_the_last_completed_refinement_pair(
        self,
    ) -> None:
        start_level = _build_level(2)
        end_level = _build_level(3)
        canvas = self._build_canvas(start_level)

        canvas.start_stair_placement()
        _place_point(canvas, 10.0, 20.0)
        _place_point(canvas, 40.0, 20.0)
        _set_canvas_level(canvas, end_level)
        _place_point(canvas, 50.0, 80.0)
        _place_point(canvas, 80.0, 80.0)
        _place_point(canvas, 20.0, 42.0)
        _place_point(canvas, 50.0, 42.0)
        _place_point(canvas, 25.0, 60.0)
        _place_point(canvas, 55.0, 60.0)

        self.assertEqual(
            len(
                canvas.get_stair_placement_draft()
                .intermediate_sections  # type: ignore[union-attr]
            ),
            2,
        )
        self.assertTrue(canvas.remove_last_stair_intermediate_section())
        self.assertEqual(
            len(
                canvas.get_stair_placement_draft()
                .intermediate_sections  # type: ignore[union-attr]
            ),
            1,
        )
        self.assertTrue(canvas.is_stair_ready_for_confirmation())

    def test_free_second_point_uses_wall_segment_angle_snapping(self) -> None:
        level = _build_level(2)
        canvas = self._build_canvas(level)

        canvas.start_stair_placement()
        _place_point(canvas, 20.0, 20.0)
        _place_point(canvas, 50.0, 22.0)

        pending = canvas.get_pending_stair_placement()
        self.assertIsNotNone(pending)
        self.assertAlmostEqual(pending.start_a_y, pending.start_b_y)  # type: ignore[union-attr]

    def test_coincident_second_point_is_rejected_and_first_point_is_kept(
        self,
    ) -> None:
        level = _build_level(2)
        canvas = self._build_canvas(level)
        errors: list[str] = []
        canvas.stair_placement_invalid_endpoint.connect(errors.append)

        canvas.start_stair_placement()
        _place_point(canvas, 20.0, 20.0)
        first_point = canvas.pending_stair_point
        _place_point(canvas, 20.0, 20.0)

        self.assertEqual(canvas.pending_stair_point, first_point)
        self.assertIsNone(canvas.get_pending_stair_placement())
        self.assertEqual(
            errors,
            ["The two points of a stair segment must be different."],
        )

    def test_switching_levels_mid_segment_requires_returning_to_owner_level(
        self,
    ) -> None:
        start_level = _build_level(2)
        other_level = _build_level(3)
        canvas = self._build_canvas(start_level)
        errors: list[str] = []
        canvas.stair_placement_invalid_endpoint.connect(errors.append)

        canvas.start_stair_placement()
        _place_point(canvas, 20.0, 20.0)
        first_point = canvas.pending_stair_point
        _set_canvas_level(canvas, other_level)
        _place_point(canvas, 50.0, 20.0)

        self.assertEqual(canvas.pending_stair_point, first_point)
        self.assertIsNone(canvas.get_pending_stair_placement())
        self.assertEqual(
            errors,
            [
                "Return to level 2 and place the second point of this stair "
                "segment."
            ],
        )

    def test_end_segment_must_be_on_a_different_level(self) -> None:
        level = _build_level(2)
        canvas = self._build_canvas(level)
        errors: list[str] = []
        canvas.stair_placement_invalid_endpoint.connect(errors.append)

        canvas.start_stair_placement()
        _place_point(canvas, 20.0, 20.0)
        _place_point(canvas, 50.0, 20.0)
        pending = canvas.get_pending_stair_placement()
        _place_point(canvas, 20.0, 70.0)

        self.assertEqual(canvas.get_pending_stair_placement(), pending)
        self.assertIsNone(canvas.pending_stair_point)
        self.assertEqual(
            errors,
            ["Choose a different level for the stair end segment."],
        )

    def test_escape_cancels_any_partial_stair_state(self) -> None:
        level = _build_level(2)
        canvas = self._build_canvas(level)
        cancelled: list[None] = []
        canvas.stair_placement_cancelled.connect(lambda: cancelled.append(None))

        canvas.start_stair_placement()
        _place_point(canvas, 24.0, 36.0)
        QTest.keyClick(canvas, Qt.Key.Key_Escape)

        self.assertIsNone(canvas.get_pending_stair_placement())
        self.assertIsNone(canvas.pending_stair_point)
        self.assertIsNone(canvas.pending_stair_style)
        self.assertEqual(cancelled, [None])

    def test_generic_stair_context_selects_any_of_its_four_points(self) -> None:
        level = _build_level(2)
        canvas = self._build_canvas(level)
        stairs = [_four_point_stair_dict()]
        canvas.set_stair_context(stairs, level)
        selected: list[int] = []
        deleted: list[int] = []
        canvas.stair_selected.connect(selected.append)
        canvas.stair_delete_requested.connect(deleted.append)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.AltModifier,
            pos=_image_position(canvas, 50.0, 30.0),
        )
        QTest.keyClick(canvas, Qt.Key.Key_Delete)

        self.assertEqual(selected, [0])
        self.assertEqual(deleted, [0])
        self.assertIsNone(canvas.selected_stair_index)
        self.assertEqual(canvas.stairs, stairs)

    def test_plain_click_on_a_bound_stair_point_selects_its_canvas_vertex(
        self,
    ) -> None:
        level = _build_level(2)
        vertex = level.vertex_data.add_vertex(20.0, 30.0)
        canvas = self._build_canvas(level)
        stair = _four_point_stair_dict()
        stair["start_a_vertex_id"] = vertex.id
        canvas.set_stair_context([stair], level)
        selected: list[int] = []
        canvas.stair_selected.connect(selected.append)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            pos=_image_position(canvas, vertex.x, vertex.y),
        )

        self.assertEqual(canvas.selected_vertex_id, vertex.id)
        self.assertEqual(selected, [])

    def test_intermediate_section_is_painted_and_can_select_its_stair(
        self,
    ) -> None:
        level = _build_level(2)
        canvas = self._build_canvas(level)
        stair = _four_point_stair_dict()
        stair["intermediate_sections"] = [
            {
                "level_index": 2,
                "a_x": 25.0,
                "a_y": 48.0,
                "b_x": 55.0,
                "b_y": 48.0,
            }
        ]
        canvas.set_stair_context([stair], level)
        selected: list[int] = []
        canvas.stair_selected.connect(selected.append)

        QTest.mouseClick(
            canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.AltModifier,
            pos=_image_position(canvas, 25.0, 48.0),
        )

        self.assertEqual(selected, [0])
        with (
            patch.object(canvas, "_paint_stair_segment") as paint_segment,
            patch.object(canvas, "_paint_stair_route_continuity"),
        ):
            canvas._paint_stairs(object())  # type: ignore[arg-type]
        self.assertEqual(paint_segment.call_count, 2)
        labels = [
            call.kwargs["label"] for call in paint_segment.call_args_list
        ]
        self.assertTrue(any("curve 1" in label for label in labels))

    def test_bound_point_display_and_hit_follow_move_then_use_saved_fallback(
        self,
    ) -> None:
        level = _build_level(2)
        vertex = level.vertex_data.add_vertex(20.0, 30.0)
        canvas = self._build_canvas(level)
        stair = _four_point_stair_dict()
        stair["start_a_vertex_id"] = vertex.id
        canvas.set_stair_context([stair], level)

        level.vertex_data.move_vertex(vertex.id, 78.0, 48.0)
        moved_widget_point = canvas._image_to_widget(78.0, 48.0)
        saved_widget_point = canvas._image_to_widget(20.0, 30.0)
        moved_hit = canvas._find_stair_hit(moved_widget_point)
        self.assertIsNotNone(moved_hit)
        self.assertEqual(moved_hit.endpoint_name, "start_a")  # type: ignore[union-attr]
        self.assertIsNone(canvas._find_stair_hit(saved_widget_point))

        with (
            patch.object(canvas, "_paint_stair_segment") as paint_segment,
            patch.object(canvas, "_paint_stair_route_continuity"),
        ):
            canvas._paint_stairs(object())  # type: ignore[arg-type]
        painted_moved_point = paint_segment.call_args.kwargs["point_a"]
        self.assertAlmostEqual(painted_moved_point.x(), moved_widget_point.x())
        self.assertAlmostEqual(painted_moved_point.y(), moved_widget_point.y())

        level.vertex_data.delete_vertex(vertex.id)
        fallback_hit = canvas._find_stair_hit(saved_widget_point)
        self.assertIsNotNone(fallback_hit)
        self.assertEqual(fallback_hit.endpoint_name, "start_a")  # type: ignore[union-attr]
        self.assertIsNone(canvas._find_stair_hit(moved_widget_point))

        with (
            patch.object(canvas, "_paint_stair_segment") as paint_segment,
            patch.object(canvas, "_paint_stair_route_continuity"),
        ):
            canvas._paint_stairs(object())  # type: ignore[arg-type]
        painted_fallback_point = paint_segment.call_args.kwargs["point_a"]
        self.assertAlmostEqual(painted_fallback_point.x(), saved_widget_point.x())
        self.assertAlmostEqual(painted_fallback_point.y(), saved_widget_point.y())


if __name__ == "__main__":
    unittest.main()
