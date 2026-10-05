# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from housemaker.glb import GeneratedModel
from housemaker.tour_state import (
    TourCurveData,
    TourData,
    TourFloatingTooltipData,
    TourStepData,
    TourText3DData,
)
from housemaker.viewer import (
    TOUR_CAMERA_TARGET_COLOR,
    TOUR_DRAFT_OVERLAY_ID,
    TOUR_EDIT_TARGET_ACTION,
    TOUR_EDIT_TARGET_ACTION_ROTATION,
    TOUR_EDIT_TARGET_CAMERA_TARGET,
    TOUR_EDIT_TARGET_CURVE_POINT,
    TOUR_EDIT_TARGET_TRIGGER,
    TOUR_NEXT_CAMERA_TARGET_COLOR,
    TOUR_POINT_KIND_CURVE,
    TOUR_POINT_KIND_TRIGGER,
    TOUR_STEP_COLOR,
    TOUR_TIMELINE_PROGRESS_COLOR,
    TOUR_UNSELECTED_OPACITY,
    GlbViewerWidget,
    evaluate_open_catmull_rom_curve,
    sample_open_catmull_rom_curve,
)


# ### Fixture helpers ###
def _build_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(8.0, 8.0, 0.2))
    return GeneratedModel(mesh=mesh, scene=trimesh.Scene(mesh), glb_bytes=b"")


def _build_tour(
    tour_id: str,
    *,
    y: float = 0.0,
    steps: tuple[TourStepData, ...] = (),
    duration_seconds: float = 10.0,
) -> TourData:
    return TourData(
        tour_id=tour_id,
        name=tour_id.title(),
        trigger_point=(0.0, y, 0.0),
        curve=TourCurveData(
            points=((0.0, y, 2.0), (10.0, y, 2.0)),
        ),
        steps=steps,
        duration_seconds=duration_seconds,
    )


def _tour_pick(
    tour_id: str,
    kind: str,
    reference: object | None = None,
    step_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        tour_id=tour_id,
        kind=kind,
        reference=reference,
        step_id=step_id,
    )


# ### Tests ###
class TourCurveViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_open_curve_sampling_preserves_endpoints_and_linear_two_point_path(
        self,
    ) -> None:
        points = ((0.0, 0.0, 1.6), (10.0, 0.0, 1.6))

        self.assertEqual(
            evaluate_open_catmull_rom_curve(points, 0.5),
            (5.0, 0.0, 1.6),
        )
        samples = sample_open_catmull_rom_curve(points, sample_count=5)
        np.testing.assert_allclose(samples[0], points[0])
        np.testing.assert_allclose(samples[-1], points[-1])

    def test_curve_placement_repeats_and_finishes_with_enter(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_model(_build_model())
        placed_spy = QSignalSpy(viewer.tour_point_placed)
        finished_spy = QSignalSpy(viewer.tour_curve_finished)

        self.assertTrue(
            viewer.begin_tour_point_placement(
                TOUR_POINT_KIND_CURVE,
                vertical_offset_meters=1.6,
            )
        )
        viewer._tour_point_hover_position = (0.0, 0.0, 1.6)
        self.assertTrue(viewer._commit_tour_point_placement())
        viewer._tour_point_hover_position = (4.0, 0.0, 1.6)
        self.assertTrue(viewer._commit_tour_point_placement())

        self.assertEqual(placed_spy.count(), 2)
        self.assertTrue(viewer.is_tour_point_placement_active)
        QTest.keyClick(viewer.view, Qt.Key.Key_Return)
        self.assertEqual(finished_spy.count(), 1)
        self.assertFalse(viewer.is_tour_point_placement_active)

    def test_non_curve_placement_does_not_consume_finish_shortcuts(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_model(_build_model())
        finish_spy = QSignalSpy(viewer.view.primary_pointer_finish_requested)

        self.assertTrue(viewer.begin_tour_point_placement(TOUR_POINT_KIND_TRIGGER))
        QTest.keyClick(viewer.view, Qt.Key.Key_Return)

        self.assertEqual(finish_spy.count(), 0)
        self.assertTrue(viewer.is_tour_point_placement_active)

    def test_fallback_plane_allows_point_placement_in_an_empty_scene(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        self.assertTrue(
            viewer.begin_tour_point_placement(
                TOUR_POINT_KIND_TRIGGER,
                fallback_plane_z=2.0,
            )
        )
        point = viewer._pick_tour_world_point(
            np.asarray((1.0, 3.0, 10.0)),
            np.asarray((0.0, 0.0, -1.0)),
        )

        np.testing.assert_allclose(point, (1.0, 3.0, 2.0), atol=1e-6)

    def test_trigger_drag_emits_center_and_rectangular_area_size(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        placed_spy = QSignalSpy(viewer.tour_point_placed)
        self.assertTrue(
            viewer.begin_tour_point_placement(
                TOUR_POINT_KIND_TRIGGER,
                fallback_plane_z=0.0,
            )
        )
        viewer._tour_trigger_drag_start = (1.0, 2.0, 0.0)
        viewer._tour_point_hover_position = (5.0, 8.0, 0.0)

        self.assertTrue(viewer._commit_tour_point_placement())

        self.assertEqual(placed_spy.count(), 1)
        self.assertEqual(placed_spy.at(0)[0], TOUR_POINT_KIND_TRIGGER)
        self.assertEqual(
            placed_spy.at(0)[1],
            {
                "center": (3.0, 5.0, 0.0),
                "size": (4.0, 6.0),
            },
        )

    def test_trigger_area_renders_and_is_pickable_away_from_its_center(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        tour = replace(_build_tour("tour"), trigger_area_size=(4.0, 6.0))
        viewer.set_tour_overlays((tour,))

        item = viewer._tour_trigger_overlay_items["tour"]
        self.assertEqual(item._housemaker_tour_trigger_size, (4.0, 6.0))
        np.testing.assert_allclose(
            item._housemaker_tour_trigger_bounds[0],
            (-2.0, -3.0, 0.0),
        )
        with (
            patch.object(
                viewer.view,
                "build_camera_ray",
                return_value=(
                    np.asarray((1.8, 2.8, 5.0)),
                    np.asarray((0.0, 0.0, -1.0)),
                ),
            ),
            patch(
                "housemaker.viewer._project_vertices_to_view",
                side_effect=lambda vertices, *_args: np.column_stack(
                    (
                        np.zeros(len(np.asarray(vertices))),
                        np.zeros(len(np.asarray(vertices))),
                        np.zeros(len(np.asarray(vertices))),
                        np.zeros(len(np.asarray(vertices))),
                    )
                ),
            ),
        ):
            picked = viewer._pick_tour_overlay(QPointF(500.0, 500.0))

        self.assertIsNotNone(picked)
        self.assertEqual(picked.kind, TOUR_EDIT_TARGET_TRIGGER)

    def test_curve_point_picker_applies_eye_height_after_nearest_scene_hit(
        self,
    ) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_model(_build_model())
        viewer.begin_tour_point_placement(
            TOUR_POINT_KIND_CURVE,
            vertical_offset_meters=1.6,
        )

        point = viewer._pick_tour_world_point(
            np.asarray((0.0, 0.0, 10.0)),
            np.asarray((0.0, 0.0, -1.0)),
        )

        self.assertIsNotNone(point)
        np.testing.assert_allclose(point, (0.0, 0.0, 1.7), atol=1e-6)

    def test_multi_tour_opacity_has_targets_and_shows_red_progress(self) -> None:
        action = TourText3DData(
            component_id="text-1",
            text="Welcome",
            position=(5.0, 2.0, 2.2),
            size_points=22.0,
        )
        step = TourStepData(
            step_id="step-1",
            progress=0.0,
            camera_target=(0.0, 1.0, 2.0),
            actions=(action,),
        )
        first = _build_tour("first", steps=(step,))
        second = _build_tour("second", y=3.0)
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (first, second),
            selected_tour_id="first",
            selected_step_id="step-1",
            selected_action_id="text-1",
            timeline_progress=0.5,
        )

        self.assertEqual(set(viewer._tour_curve_overlay_items), {"first", "second"})
        self.assertEqual(viewer._tour_curve_overlay_items["first"].color[3], 1.0)
        self.assertEqual(
            viewer._tour_curve_overlay_items["second"].color[3],
            TOUR_UNSELECTED_OPACITY,
        )
        roles = {
            getattr(item, "_housemaker_tour_role", None)
            for item in viewer._tour_static_overlay_items
        }
        self.assertIn("camera_target", roles)
        self.assertIn(("first", "step-1"), viewer._tour_camera_target_overlay_items)
        self.assertEqual(len(viewer._tour_preview_overlay_items), 2)
        progress_item = viewer._tour_preview_overlay_items[0]
        self.assertEqual(progress_item.color, TOUR_TIMELINE_PROGRESS_COLOR)
        np.testing.assert_allclose(progress_item.pos[0], (5.0, 0.0, 2.0))
        direction_item = viewer._tour_preview_overlay_items[1]
        self.assertEqual(
            direction_item._housemaker_tour_role,
            "timeline_direction",
        )
        self.assertEqual(direction_item.color, TOUR_TIMELINE_PROGRESS_COLOR)
        self.assertEqual(
            [item.text for item in viewer._tour_text_overlay_items], ["Welcome"]
        )
        text_item = viewer._tour_text_overlay_items[0]
        self.assertTrue(text_item._housemaker_is_extruded_text)
        self.assertGreater(len(text_item.opts["meshdata"].vertexes()), 8)

        static_items = tuple(viewer._tour_static_overlay_items)
        self.assertTrue(viewer.set_tour_timeline_progress(0.75))
        self.assertEqual(tuple(viewer._tour_static_overlay_items), static_items)
        np.testing.assert_allclose(
            viewer._tour_preview_overlay_items[0].pos[0],
            (7.5, 0.0, 2.0),
        )

    def test_step_markers_are_batched_per_tour_and_follow_step_progress(
        self,
    ) -> None:
        first_steps = (
            TourStepData(
                step_id="first-quarter",
                progress=0.25,
                camera_target=(2.5, 1.0, 2.0),
            ),
            TourStepData(
                step_id="first-three-quarters",
                progress=0.75,
                camera_target=(7.5, 1.0, 2.0),
            ),
        )
        second_step = TourStepData(
            step_id="second-middle",
            progress=0.5,
            camera_target=(5.0, 4.0, 2.0),
        )
        first = _build_tour("first", steps=first_steps)
        second = _build_tour("second", y=3.0, steps=(second_step,))
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (first, second),
            selected_tour_id=first.tour_id,
        )

        self.assertEqual(set(viewer._tour_step_overlay_items), {"first", "second"})
        first_item = viewer._tour_step_overlay_items["first"]
        second_item = viewer._tour_step_overlay_items["second"]
        np.testing.assert_allclose(
            first_item.pos,
            ((2.5, 0.0, 2.0), (7.5, 0.0, 2.0)),
        )
        np.testing.assert_allclose(second_item.pos, ((5.0, 3.0, 2.0),))
        self.assertEqual(first_item.color, TOUR_STEP_COLOR)
        self.assertEqual(
            second_item.color,
            (*TOUR_STEP_COLOR[:3], TOUR_UNSELECTED_OPACITY),
        )
        self.assertEqual(
            first_item._housemaker_step_ids,
            ("first-quarter", "first-three-quarters"),
        )
        self.assertEqual(
            sum(
                getattr(item, "_housemaker_tour_role", None) == "step_markers"
                for item in viewer._tour_static_overlay_items
            ),
            2,
        )

        updated_first = replace(
            first,
            steps=(replace(first_steps[0], progress=0.4), first_steps[1]),
        )
        viewer.set_tour_overlays(
            (updated_first, second),
            selected_tour_id=first.tour_id,
        )

        np.testing.assert_allclose(
            viewer._tour_step_overlay_items["first"].pos,
            ((4.0, 0.0, 2.0), (7.5, 0.0, 2.0)),
        )

    def test_red_timeline_arrow_tracks_interpolated_and_live_camera_target(
        self,
    ) -> None:
        steps = (
            TourStepData(
                step_id="step-1",
                progress=0.0,
                camera_target=(0.0, 3.0, 2.0),
            ),
            TourStepData(
                step_id="step-2",
                progress=1.0,
                camera_target=(10.0, 3.0, 2.0),
            ),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour", steps=steps),),
            selected_tour_id="tour",
            selected_step_id="step-1",
            timeline_progress=0.5,
        )

        direction_item = next(
            item
            for item in viewer._tour_preview_overlay_items
            if item._housemaker_tour_role == "timeline_direction"
        )
        np.testing.assert_allclose(direction_item.pos[0], (5.0, 0.0, 2.0))
        np.testing.assert_allclose(
            direction_item._housemaker_tour_direction,
            (0.0, 1.0, 0.0),
            atol=1e-7,
        )

        self.assertTrue(viewer.set_tour_preview(0.5, (5.0, -4.0, 2.0)))

        live_direction_item = next(
            item
            for item in viewer._tour_preview_overlay_items
            if item._housemaker_tour_role == "timeline_direction"
        )
        np.testing.assert_allclose(
            live_direction_item._housemaker_tour_direction,
            (0.0, -1.0, 0.0),
            atol=1e-7,
        )

        self.assertTrue(
            viewer.set_tour_timeline_progress(
                0.5,
                camera_position=(5.0, 5.0, 2.0),
                camera_target=(5.0, 6.0, 2.0),
            )
        )
        timeline_direction_item = next(
            item
            for item in viewer._tour_preview_overlay_items
            if item._housemaker_tour_role == "timeline_direction"
        )
        np.testing.assert_allclose(
            timeline_direction_item.pos[0],
            (5.0, 5.0, 2.0),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            timeline_direction_item._housemaker_tour_direction,
            (0.0, 1.0, 0.0),
            atol=1e-7,
        )

        self.assertTrue(viewer.set_tour_timeline_progress(0.5))
        np.testing.assert_allclose(
            viewer._tour_preview_overlay_items[0].pos[0],
            (5.0, 0.0, 2.0),
            atol=1e-7,
        )

    def test_tour_preview_accepts_an_explicit_idle_camera_position(self) -> None:
        viewer = GlbViewerWidget(tour_direction_arrow_enabled=False)
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour"),),
            selected_tour_id="tour",
            timeline_progress=0.5,
        )

        self.assertTrue(
            viewer.set_tour_preview(
                0.5,
                (5.0, 2.0, 2.0),
                camera_position=(5.0, -3.0, 2.5),
            )
        )

        camera_position = viewer.view.cameraPosition()
        np.testing.assert_allclose(
            (camera_position.x(), camera_position.y(), camera_position.z()),
            (5.0, -3.0, 2.5),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            viewer._tour_preview_overlay_items[0].pos[0],
            (5.0, 0.0, 2.0),
            atol=1e-6,
        )

    def test_direction_arrow_can_be_hidden_without_hiding_progress_sphere(
        self,
    ) -> None:
        viewer = GlbViewerWidget(tour_direction_arrow_enabled=False)
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour"),),
            selected_tour_id="tour",
            timeline_progress=0.5,
        )

        roles = {
            item._housemaker_tour_role
            for item in viewer._tour_preview_overlay_items
        }

        self.assertEqual(roles, {"timeline_progress"})

    def test_red_timeline_arrow_uses_curve_tangent_for_coincident_target(
        self,
    ) -> None:
        steps = (
            TourStepData(
                step_id="step-1",
                progress=0.0,
                camera_target=(0.0, 0.0, 2.0),
            ),
            TourStepData(
                step_id="step-2",
                progress=1.0,
                camera_target=(10.0, 0.0, 2.0),
            ),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour", steps=steps),),
            selected_tour_id="tour",
            selected_step_id="step-1",
            timeline_progress=0.5,
        )

        direction_item = next(
            item
            for item in viewer._tour_preview_overlay_items
            if item._housemaker_tour_role == "timeline_direction"
        )

        np.testing.assert_allclose(
            direction_item._housemaker_tour_direction,
            (1.0, 0.0, 0.0),
            atol=1e-7,
        )

    def test_draft_is_rendered_alongside_saved_tours(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (_build_tour("saved"),),
            selected_tour_id=TOUR_DRAFT_OVERLAY_ID,
            draft={
                "trigger_point": (2.0, 2.0, 0.0),
                "curve_points": ((2.0, 2.0, 2.0), (4.0, 2.0, 2.0)),
            },
        )

        self.assertEqual(
            set(viewer._tour_curve_overlay_items),
            {"saved", TOUR_DRAFT_OVERLAY_ID},
        )
        self.assertEqual(viewer._tour_selected_tour_id, TOUR_DRAFT_OVERLAY_ID)
        self.assertIsNotNone(viewer._tour_control_point_overlay_item)

    def test_selected_curve_click_toggles_selection_and_hides_actions(self) -> None:
        step = TourStepData(
            step_id="step-1",
            progress=0.0,
            camera_target=(1.0, 0.0, 2.0),
            actions=(
                TourText3DData(
                    component_id="text-1",
                    text="Visible",
                    position=(1.0, 1.0, 2.0),
                ),
            ),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id="step-1",
        )
        selection_spy = QSignalSpy(viewer.tour_selection_changed)
        self.assertEqual(len(viewer._tour_text_overlay_items), 1)

        viewer._apply_tour_overlay_pick(_tour_pick("tour", "curve"))

        self.assertIsNone(viewer._tour_selected_tour_id)
        self.assertFalse(viewer._tour_text_overlay_items)
        self.assertFalse(viewer._tour_preview_overlay_items)
        self.assertFalse(viewer._tour_camera_target_overlay_items)
        self.assertEqual(viewer._tour_curve_overlay_items["tour"].color[3], 0.25)
        self.assertEqual(selection_spy.count(), 1)
        self.assertIsNone(selection_spy.at(0)[0])

    def test_curve_and_edit_target_selection_emit_identifiers(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays((_build_tour("tour"),))
        selection_spy = QSignalSpy(viewer.tour_selection_changed)
        target_spy = QSignalSpy(viewer.tour_edit_target_selected)

        viewer._apply_tour_overlay_pick(_tour_pick("tour", "curve"))
        viewer._apply_tour_overlay_pick(
            _tour_pick("tour", TOUR_EDIT_TARGET_CURVE_POINT, 1)
        )

        self.assertEqual(selection_spy.count(), 1)
        self.assertEqual(selection_spy.at(0)[0], "tour")
        self.assertEqual(target_spy.count(), 1)
        self.assertEqual(
            tuple(target_spy.at(0)),
            ("tour", TOUR_EDIT_TARGET_CURVE_POINT, 1),
        )
        self.assertEqual(len(viewer._tour_gizmo_items), 6)

    def test_curve_point_selection_does_not_move_the_preview_camera(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour"),),
            selected_tour_id="tour",
        )
        camera_before = viewer.view.cameraPosition()
        center_before = viewer.view.opts["center"]

        viewer._apply_tour_overlay_pick(
            _tour_pick("tour", TOUR_EDIT_TARGET_CURVE_POINT, 0)
        )

        camera_after = viewer.view.cameraPosition()
        center_after = viewer.view.opts["center"]
        np.testing.assert_allclose(
            (camera_after.x(), camera_after.y(), camera_after.z()),
            (camera_before.x(), camera_before.y(), camera_before.z()),
        )
        np.testing.assert_allclose(
            (center_after.x(), center_after.y(), center_after.z()),
            (center_before.x(), center_before.y(), center_before.z()),
        )

    def test_camera_target_sphere_selects_and_moves_with_the_gizmo(self) -> None:
        step = TourStepData(
            step_id="step-1",
            progress=0.0,
            camera_target=(0.0, 1.0, 2.0),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id="step-1",
        )
        target_spy = QSignalSpy(viewer.tour_edit_target_selected)
        finished_spy = QSignalSpy(viewer.tour_edit_finished)

        viewer._apply_tour_overlay_pick(
            _tour_pick(
                "tour",
                TOUR_EDIT_TARGET_CAMERA_TARGET,
                "step-1",
                "step-1",
            )
        )

        self.assertEqual(tuple(target_spy.at(0)), ("tour", "camera_target", "step-1"))
        self.assertEqual(len(viewer._tour_gizmo_items), 6)
        direction = np.asarray((0.0, 1.0, -1.0), dtype=float)
        direction /= np.linalg.norm(direction)
        rays = iter(
            (
                (np.asarray((0.0, -9.0, 12.0)), direction),
                (np.asarray((2.0, -9.0, 12.0)), direction),
            )
        )
        with patch.object(
            viewer.view,
            "build_camera_ray",
            side_effect=lambda _position: next(rays),
        ):
            self.assertTrue(viewer._begin_tour_edit_drag(0, QPointF()))
            self.assertTrue(viewer._finish_tour_edit_drag(QPointF()))

        self.assertEqual(finished_spy.count(), 1)
        self.assertEqual(finished_spy.at(0)[:3], ["tour", "camera_target", "step-1"])
        np.testing.assert_allclose(finished_spy.at(0)[3], (2.0, 1.0, 2.0))

    def test_camera_target_drag_uses_its_start_camera_after_live_updates(
        self,
    ) -> None:
        step = TourStepData(
            step_id="step-1",
            progress=0.0,
            camera_target=(0.0, 1.0, 2.0),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.resize(640, 480)
        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id="step-1",
            timeline_progress=0.0,
        )
        self.assertTrue(viewer.set_tour_preview(0.0, step.camera_target))
        self.assertTrue(
            viewer.select_tour_edit_target(
                "tour",
                TOUR_EDIT_TARGET_CAMERA_TARGET,
                "step-1",
            )
        )
        start_pointer = QPointF(viewer.view.width() * 0.5, viewer.view.height() * 0.5)
        self.assertTrue(viewer._begin_tour_edit_drag(0, start_pointer))
        x_positions = [step.camera_target[0]]

        for pixel_offset in (5.0, 10.0, 15.0, 20.0, 25.0):
            self.assertTrue(
                viewer._update_tour_edit_drag(
                    QPointF(start_pointer.x() + pixel_offset, start_pointer.y())
                )
            )
            drag = viewer._tour_edit_drag
            self.assertIsNotNone(drag)
            next_target = drag.preview_position
            x_positions.append(next_target[0])
            self.assertTrue(viewer.set_tour_preview(0.0, next_target))

        increments = np.diff(x_positions)
        np.testing.assert_allclose(
            increments,
            np.full_like(increments, increments[0]),
            rtol=1e-5,
            atol=1e-7,
        )
        viewer._cancel_tour_edit_drag(notify=False)

    def test_curve_point_gizmo_previews_and_commits_on_release(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour"),),
            selected_tour_id="tour",
        )
        viewer._apply_tour_overlay_pick(
            _tour_pick("tour", TOUR_EDIT_TARGET_CURVE_POINT, 0)
        )
        preview_spy = QSignalSpy(viewer.tour_edit_preview_changed)
        finished_spy = QSignalSpy(viewer.tour_edit_finished)
        direction = np.asarray((0.0, 1.0, -1.0), dtype=float)
        direction /= np.linalg.norm(direction)
        rays = iter(
            (
                (np.asarray((0.0, -10.0, 12.0)), direction),
                (np.asarray((2.0, -10.0, 12.0)), direction),
            )
        )

        with patch.object(
            viewer.view, "build_camera_ray", side_effect=lambda _p: next(rays)
        ):
            self.assertTrue(viewer._begin_tour_edit_drag(0, QPointF()))
            self.assertTrue(viewer._finish_tour_edit_drag(QPointF()))

        self.assertEqual(preview_spy.count(), 1)
        self.assertEqual(preview_spy.at(0)[:3], ["tour", "curve_point", 0])
        np.testing.assert_allclose(preview_spy.at(0)[3], (2.0, 0.0, 2.0))
        self.assertEqual(finished_spy.count(), 1)
        self.assertTrue(finished_spy.at(0)[4])
        np.testing.assert_allclose(
            viewer._get_tour_curve_display_points(viewer._tour_overlays["tour"])[0],
            (2.0, 0.0, 2.0),
        )

    def test_cancelled_gizmo_drag_clears_its_live_preview(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour"),),
            selected_tour_id="tour",
        )
        viewer._apply_tour_overlay_pick(
            _tour_pick("tour", TOUR_EDIT_TARGET_CURVE_POINT, 0)
        )
        finished_spy = QSignalSpy(viewer.tour_edit_finished)
        direction = np.asarray((0.0, 1.0, -1.0), dtype=float)
        direction /= np.linalg.norm(direction)
        rays = iter(
            (
                (np.asarray((0.0, -10.0, 12.0)), direction),
                (np.asarray((2.0, -10.0, 12.0)), direction),
            )
        )

        with patch.object(
            viewer.view, "build_camera_ray", side_effect=lambda _p: next(rays)
        ):
            self.assertTrue(viewer._begin_tour_edit_drag(0, QPointF()))
            self.assertTrue(viewer._update_tour_edit_drag(QPointF()))
        self.assertTrue(viewer._tour_position_overrides)

        self.assertTrue(viewer._cancel_tour_edit_drag())

        self.assertFalse(viewer._tour_position_overrides)
        self.assertEqual(finished_spy.count(), 1)
        self.assertFalse(finished_spy.at(0)[4])
        np.testing.assert_allclose(
            viewer._get_tour_curve_display_points(viewer._tour_overlays["tour"])[0],
            (0.0, 0.0, 2.0),
        )

    def test_action_target_has_translation_and_rotation_gizmos(self) -> None:
        action = TourText3DData(
            component_id="text-1",
            text="Move me",
            position=(1.0, 2.0, 3.0),
        )
        step = TourStepData(
            step_id="step-1",
            progress=0.0,
            camera_target=(1.0, 0.0, 2.0),
            actions=(action,),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id="step-1",
            selected_action_id="text-1",
            timeline_progress=0.0,
        )

        self.assertEqual(viewer._tour_edit_target.kind, TOUR_EDIT_TARGET_ACTION)
        self.assertEqual(viewer._tour_edit_target.reference, "text-1")
        self.assertEqual(len(viewer._tour_gizmo_items), 9)
        self.assertEqual(len(viewer._tour_text_overlay_items), 1)
        self.assertEqual(viewer._tour_text_overlay_items[0]._housemaker_alpha, 1.0)
        np.testing.assert_allclose(
            viewer._get_tour_edit_target_position(viewer._tour_edit_target),
            action.position,
        )

    def test_current_and_next_camera_targets_use_distinct_pink_markers(self) -> None:
        first = TourStepData(
            step_id="first",
            progress=0.0,
            camera_target=(1.0, 0.0, 2.0),
        )
        second = TourStepData(
            step_id="second",
            progress=1.0,
            camera_target=(9.0, 0.0, 2.0),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(first, second)),),
            selected_tour_id="tour",
            selected_step_id="first",
        )

        self.assertEqual(
            set(viewer._tour_camera_target_overlay_items),
            {("tour", "first"), ("tour", "second")},
        )
        current_item = viewer._tour_camera_target_overlay_items[("tour", "first")]
        next_item = viewer._tour_camera_target_overlay_items[("tour", "second")]
        self.assertEqual(current_item.color, TOUR_CAMERA_TARGET_COLOR)
        self.assertEqual(next_item.color, TOUR_NEXT_CAMERA_TARGET_COLOR)
        self.assertEqual(current_item._housemaker_tour_target_timing, "current")
        self.assertEqual(next_item._housemaker_tour_target_timing, "next")

        def project_by_world_x(vertices: object, *_args: object) -> np.ndarray:
            points = np.asarray(vertices, dtype=float)
            return np.column_stack(
                (
                    points[:, 0] * 20.0,
                    np.full(len(points), 100.0),
                    np.zeros(len(points)),
                    np.ones(len(points)),
                )
            )

        with patch(
            "housemaker.viewer._project_vertices_to_view",
            side_effect=project_by_world_x,
        ):
            picked = viewer._pick_tour_overlay(QPointF(180.0, 100.0))

        self.assertIsNotNone(picked)
        self.assertEqual(picked.kind, TOUR_EDIT_TARGET_CAMERA_TARGET)
        self.assertEqual(picked.reference, "second")
        target_spy = QSignalSpy(viewer.tour_edit_target_selected)
        viewer._apply_tour_overlay_pick(picked)

        self.assertEqual(tuple(target_spy.at(0)), ("tour", "camera_target", "second"))
        self.assertEqual(len(viewer._tour_gizmo_items), 6)
        self.assertEqual(viewer._tour_selected_step_id, "second")
        self.assertEqual(
            set(viewer._tour_camera_target_overlay_items),
            {("tour", "second")},
        )
        self.assertEqual(
            viewer._tour_camera_target_overlay_items[("tour", "second")].color,
            TOUR_CAMERA_TARGET_COLOR,
        )

        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(first, second)),),
            selected_tour_id="tour",
            selected_step_id="first",
        )

        self.assertEqual(
            set(viewer._tour_camera_target_overlay_items),
            {("tour", "first"), ("tour", "second")},
        )
        self.assertFalse(viewer._tour_gizmo_items)

    def test_next_camera_target_uses_stable_chronological_step_order(self) -> None:
        late = TourStepData(
            step_id="late",
            progress=0.9,
            camera_target=(9.0, 0.0, 2.0),
        )
        equal_but_earlier = TourStepData(
            step_id="zz-equal-but-earlier",
            progress=0.2,
            camera_target=(1.0, 0.0, 2.0),
        )
        next_step = TourStepData(
            step_id="next",
            progress=0.6,
            camera_target=(6.0, 0.0, 2.0),
        )
        current = TourStepData(
            step_id="current",
            progress=0.2,
            camera_target=(2.0, 0.0, 2.0),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (
                _build_tour(
                    "tour",
                    steps=(late, equal_but_earlier, current, next_step),
                ),
            ),
            selected_tour_id="tour",
            selected_step_id="current",
        )

        self.assertEqual(
            set(viewer._tour_camera_target_overlay_items),
            {("tour", "current"), ("tour", "next")},
        )

    def test_current_camera_target_renders_last_when_targets_overlap(self) -> None:
        shared_target = (5.0, 2.0, 2.0)
        first = TourStepData(
            step_id="first",
            progress=0.0,
            camera_target=shared_target,
        )
        second = TourStepData(
            step_id="second",
            progress=1.0,
            camera_target=shared_target,
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(first, second)),),
            selected_tour_id="tour",
            selected_step_id="first",
        )

        rendered_target_timings = [
            item._housemaker_tour_target_timing
            for item in viewer._tour_static_overlay_items
            if getattr(item, "_housemaker_tour_role", None)
            == TOUR_EDIT_TARGET_CAMERA_TARGET
        ]
        self.assertEqual(rendered_target_timings, ["next", "current"])

    def test_text_rotation_ring_previews_and_commits_xyz_degrees(self) -> None:
        action = TourText3DData(
            component_id="text-1",
            text="Rotate me",
            position=(1.0, 2.0, 3.0),
            rotation_degrees=(10.0, 20.0, 30.0),
        )
        step = TourStepData(
            step_id="step-1",
            progress=0.0,
            camera_target=(1.0, 0.0, 2.0),
            actions=(action,),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id="step-1",
            selected_action_id="text-1",
        )
        preview_spy = QSignalSpy(viewer.tour_edit_preview_changed)
        finished_spy = QSignalSpy(viewer.tour_edit_finished)
        rays = iter(
            (
                (
                    np.asarray((2.0, 2.0, 10.0)),
                    np.asarray((0.0, 0.0, -1.0)),
                ),
                (
                    np.asarray((1.0, 3.0, 10.0)),
                    np.asarray((0.0, 0.0, -1.0)),
                ),
            )
        )

        with patch.object(
            viewer.view,
            "build_camera_ray",
            side_effect=lambda _position: next(rays),
        ):
            self.assertTrue(viewer._begin_tour_rotation_drag(2, QPointF()))
            self.assertTrue(viewer._finish_tour_edit_drag(QPointF()))

        self.assertEqual(preview_spy.count(), 1)
        self.assertEqual(
            preview_spy.at(0)[:3],
            ["tour", TOUR_EDIT_TARGET_ACTION_ROTATION, "text-1"],
        )
        np.testing.assert_allclose(preview_spy.at(0)[3], (10.0, 20.0, 120.0))
        self.assertEqual(finished_spy.count(), 1)
        self.assertTrue(finished_spy.at(0)[4])

    def test_text_invisible_hit_box_selects_away_from_its_anchor(self) -> None:
        action = TourText3DData(
            component_id="text-1",
            text="A deliberately wide selection target",
            position=(1.0, 2.0, 3.0),
        )
        step = TourStepData(
            step_id="step-1",
            progress=0.0,
            camera_target=(1.0, 0.0, 2.0),
            actions=(action,),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id="step-1",
        )
        ray = (
            np.asarray((1.0, 2.0, 10.0)),
            np.asarray((0.0, 0.0, -1.0)),
        )

        with (
            patch.object(viewer.view, "build_camera_ray", return_value=ray),
            patch(
                "housemaker.viewer._project_vertices_to_view",
                side_effect=lambda vertices, *_args: np.tile(
                    np.asarray((100.0, 100.0, 0.0, 1.0)),
                    (len(vertices), 1),
                ),
            ),
        ):
            picked = viewer._pick_tour_overlay(QPointF(0.0, 0.0))

        self.assertIsNotNone(picked)
        self.assertEqual(picked.kind, TOUR_EDIT_TARGET_ACTION)
        self.assertEqual(picked.reference, "text-1")

    def test_camera_target_wins_over_an_overlapping_text_hit_box(self) -> None:
        target = (1.0, 2.0, 3.0)
        action = TourText3DData(
            component_id="text-1",
            text="Wide text overlapping the camera target",
            position=target,
        )
        step = TourStepData(
            step_id="step-1",
            progress=0.0,
            camera_target=target,
            actions=(action,),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id="step-1",
        )
        ray = (
            np.asarray((1.0, 2.0, 10.0)),
            np.asarray((0.0, 0.0, -1.0)),
        )

        with (
            patch.object(viewer.view, "build_camera_ray", return_value=ray),
            patch(
                "housemaker.viewer._project_vertices_to_view",
                side_effect=lambda vertices, *_args: np.tile(
                    np.asarray((100.0, 100.0, 0.0, 1.0)),
                    (len(vertices), 1),
                ),
            ),
        ):
            picked = viewer._pick_tour_overlay(QPointF(100.0, 100.0))

        self.assertIsNotNone(picked)
        self.assertEqual(picked.kind, TOUR_EDIT_TARGET_CAMERA_TARGET)
        self.assertEqual(picked.reference, "step-1")

    def test_current_authored_step_shows_all_visual_actions(self) -> None:
        step = TourStepData(
            step_id="step-1",
            progress=0.4,
            camera_target=(5.0, 1.0, 2.0),
            actions=(
                TourText3DData(
                    component_id="selected-text",
                    text="Selected",
                    position=(4.0, 1.0, 2.0),
                    fade_delay_ms=5000.0,
                ),
                TourText3DData(
                    component_id="other-text",
                    text="Also visible",
                    position=(4.0, 2.0, 2.0),
                    fade_delay_ms=5000.0,
                ),
            ),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id="step-1",
            selected_action_id="selected-text",
            timeline_progress=0.4,
        )

        self.assertEqual(
            {item.text for item in viewer._tour_text_overlay_items},
            {"Selected", "Also visible"},
        )
        self.assertTrue(
            all(
                item._housemaker_alpha == 1.0
                for item in viewer._tour_text_overlay_items
            )
        )

    def test_external_tour_edit_preview_updates_retained_visual_position(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour"),),
            selected_tour_id="tour",
        )

        self.assertTrue(
            viewer.set_tour_edit_preview_position(
                "tour",
                TOUR_EDIT_TARGET_CURVE_POINT,
                0,
                (2.0, 3.0, 4.0),
            )
        )

        np.testing.assert_allclose(
            viewer._get_tour_curve_display_points(
                viewer._tour_overlays["tour"]
            )[0],
            (2.0, 3.0, 4.0),
        )

    def test_selected_step_wins_when_steps_share_the_same_progress(self) -> None:
        first = TourStepData(
            step_id="first",
            progress=0.5,
            camera_target=(6.0, 0.0, 2.0),
            actions=(
                TourText3DData(
                    component_id="first-text",
                    text="First",
                    position=(1.0, 0.0, 2.0),
                ),
            ),
        )
        second = TourStepData(
            step_id="second",
            progress=0.5,
            camera_target=(6.0, 0.0, 2.0),
            actions=(
                TourText3DData(
                    component_id="second-text",
                    text="Second",
                    position=(2.0, 0.0, 2.0),
                ),
            ),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(first, second)),),
            selected_tour_id="tour",
            selected_step_id=first.step_id,
            selected_action_id="first-text",
            timeline_progress=0.5,
        )

        self.assertEqual(
            [item.text for item in viewer._tour_text_overlay_items],
            ["First"],
        )

    def test_text_alpha_uses_fade_delay_duration_and_previous_step_fade(self) -> None:
        first = TourStepData(
            step_id="first",
            progress=0.0,
            camera_target=(1.0, 0.0, 2.0),
            actions=(
                TourText3DData(
                    component_id="first-text",
                    text="First",
                    position=(1.0, 0.0, 2.0),
                    fade_duration_ms=2000.0,
                ),
            ),
        )
        second = TourStepData(
            step_id="second",
            progress=0.2,
            camera_target=(3.0, 0.0, 2.0),
            actions=(
                TourText3DData(
                    component_id="second-text",
                    text="Second",
                    position=(2.0, 0.0, 2.0),
                    fade_delay_ms=500.0,
                    fade_duration_ms=1000.0,
                ),
            ),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(first, second)),),
            selected_tour_id="tour",
            selected_step_id="second",
            timeline_progress=0.3,
        )

        alphas = {
            item.text: item._housemaker_alpha
            for item in viewer._tour_text_overlay_items
        }
        self.assertAlmostEqual(alphas["First"], 0.5)
        self.assertAlmostEqual(alphas["Second"], 0.5)

    def test_camera_direction_and_screen_center_hit_or_fallback(self) -> None:
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)
        viewer.set_tour_overlays(
            (_build_tour("tour"),),
            selected_tour_id="tour",
        )

        self.assertTrue(viewer.set_tour_preview(0.5, (5.0, 3.0, 2.0)))
        np.testing.assert_allclose(
            viewer.get_tour_camera_look_direction(),
            (0.0, 1.0, 0.0),
            atol=1e-6,
        )
        camera_position = viewer.view.cameraPosition()
        np.testing.assert_allclose(
            (camera_position.x(), camera_position.y(), camera_position.z()),
            (5.0, 0.0, 2.0),
            atol=1e-6,
        )
        forward_point = viewer.get_camera_forward_world_point()
        np.testing.assert_allclose(forward_point, (5.0, 1.0, 2.0), atol=1e-6)
        self.assertAlmostEqual(
            float(np.linalg.norm(np.asarray(forward_point) - (5.0, 0.0, 2.0))),
            1.0,
        )
        with self.assertRaises(ValueError):
            viewer.get_camera_forward_world_point(0.0)

        ray = (
            np.asarray((1.0, 2.0, 3.0)),
            np.asarray((0.0, 0.0, -1.0)),
        )
        with (
            patch.object(viewer.view, "build_camera_ray", return_value=ray),
            patch.object(
                viewer,
                "_pick_nearest_tour_scene_hit",
                return_value=(np.asarray((4.0, 5.0, 6.0)), 3.0),
            ),
        ):
            self.assertEqual(
                viewer.get_tour_screen_center_world_point(),
                (4.0, 5.0, 6.0),
            )
        with (
            patch.object(viewer.view, "build_camera_ray", return_value=ray),
            patch.object(viewer, "_pick_nearest_tour_scene_hit", return_value=None),
        ):
            self.assertEqual(
                viewer.get_tour_screen_center_world_point(fallback_distance_meters=5.0),
                (1.0, 2.0, -2.0),
            )

    def test_floating_tooltip_hotspot_hovers_as_html_in_tour_preview(self) -> None:
        tooltip = TourFloatingTooltipData(
            component_id="tooltip-1",
            anchor_point=(1.0, 2.0, 1.5),
            tooltip_position="opposite",
            html_body="<h1>Kitchen</h1><img src='kitchen.png'>",
            style="h1 { color: #ffcc00; }",
        )
        step = TourStepData(
            step_id="tooltip-step",
            progress=0.0,
            camera_target=(0.0, 1.0, 2.0),
            actions=(tooltip,),
        )
        viewer = GlbViewerWidget(tour_html_tooltips_enabled=True)
        self.addCleanup(viewer.close)
        viewer.resize(800, 600)
        viewer.show()
        self.app.processEvents()

        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id=step.step_id,
            selected_action_id=tooltip.component_id,
        )

        self.assertEqual(len(viewer._tour_tooltip_hotspot_items), 1)
        self.assertEqual(len(viewer._tour_gizmo_items), 6)
        self.assertFalse(
            viewer.set_tour_edit_preview_rotation(
                "tour",
                tooltip.component_id,
                (0.0, 10.0, 0.0),
            )
        )
        with patch.object(
            viewer,
            "_project_tour_world_point",
            return_value=(QPointF(100.0, 300.0), 0.5),
        ):
            viewer._update_tour_tooltip_hover(QPointF(100.0, 300.0))

        self.assertEqual(
            viewer._tour_hovered_tooltip_action_id,
            tooltip.component_id,
        )
        self.assertIsNotNone(viewer.tour_tooltip_overlay)
        assert viewer.tour_tooltip_overlay is not None
        self.assertTrue(viewer.tour_tooltip_overlay.isVisible())
        rendered_html = viewer.tour_tooltip_overlay.toHtml().casefold()
        self.assertIn("kitchen", rendered_html)
        self.assertIn("#ffcc00", rendered_html)

    def test_floating_tooltip_hotspot_remains_in_scene_without_html_overlay(
        self,
    ) -> None:
        tooltip = TourFloatingTooltipData(
            component_id="scene-tooltip",
            anchor_point=(2.0, 1.0, 1.5),
        )
        step = TourStepData(
            step_id="scene-step",
            progress=0.0,
            camera_target=(0.0, 1.0, 2.0),
            actions=(tooltip,),
        )
        viewer = GlbViewerWidget()
        self.addCleanup(viewer.close)

        viewer.set_tour_overlays(
            (_build_tour("tour", steps=(step,)),),
            selected_tour_id="tour",
            selected_step_id=step.step_id,
        )

        self.assertEqual(len(viewer._tour_tooltip_hotspot_items), 1)
        self.assertIsNone(viewer.tour_tooltip_overlay)


# ### Entry point ###
if __name__ == "__main__":
    unittest.main()
