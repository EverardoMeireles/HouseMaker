# ### Environment setup ###
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QLabel

from housemaker.canvas_openings import (
    CANVAS_OPENING_WINDOW,
    CanvasOpeningBounds,
    CanvasOpeningReference,
    CanvasOpeningTarget,
)
from housemaker.level_coordinates import level_world_to_image_xy
from housemaker.main import BlueprintWorkspace, _build_local_file_revision
from housemaker.models import DoorwayData, LevelData, OpenSpaceData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
class _VerificationHarness:
    """Exercise context and label helpers without constructing the 3D viewers."""

    _wall_verification_context_signature = (
        BlueprintWorkspace._wall_verification_context_signature
    )
    _wall_verification_context_is_current = (
        BlueprintWorkspace._wall_verification_context_is_current
    )
    _wall_verification_excluded_polygons = (
        BlueprintWorkspace._wall_verification_excluded_polygons
    )
    _invalidate_stale_wall_verification = (
        BlueprintWorkspace._invalidate_stale_wall_verification
    )
    _update_wall_verification_instructions = (
        BlueprintWorkspace._update_wall_verification_instructions
    )
    _animate_wall_verification_instructions = (
        BlueprintWorkspace._animate_wall_verification_instructions
    )

    def __init__(self) -> None:
        self.current_level = LevelData(2, "Ground")
        self.stairs = []
        self._wall_verification_preview = None
        self._wall_verification_active_job_id = None
        self._wall_verification_runtimes = {}
        self._canvas_opening_targets_by_key = {}
        self._generation_settings = SimpleNamespace(
            include_llm_wall_verification=False
        )
        self._is_shutdown = False
        self._wall_verification_instruction_phase = 0
        self.wall_verification_instructions_label = QLabel()
        self._wall_verification_instruction_timer = QTimer()
        self._wall_verification_instruction_timer.setInterval(80)
        self.dismissed_messages = []

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(
            level_index=self.current_level.index,
            source_revision=_build_local_file_revision(self.current_level.image_path),
            context_signature=self._wall_verification_context_signature(
                self.current_level
            ),
        )

    def _dismiss_wall_verification(self, *, message: str) -> None:
        self.dismissed_messages.append(message)


def _window_target(level_index: int) -> CanvasOpeningTarget:
    return CanvasOpeningTarget(
        reference=CanvasOpeningReference(
            CANVAS_OPENING_WINDOW, level_index, 0, f"window-{level_index}"
        ),
        wall_surface_id=f"level-{level_index}-wall",
        plane_start_world=(0.0, 0.0, 0.0),
        wall_tangent_world=(1.0, 0.0, 0.0),
        wall_normal_world=(0.0, -1.0, 0.0),
        wall_width_meters=4.0,
        wall_height_meters=3.0,
        minimum_width_meters=0.1,
        minimum_height_meters=0.1,
        bounds=CanvasOpeningBounds(0.25, 0.75, 0.25, 0.75),
    )


# ### Tests ###
class WallVerificationContextMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = _VerificationHarness()

    def tearDown(self) -> None:
        self.harness._wall_verification_instruction_timer.stop()
        self.harness.wall_verification_instructions_label.deleteLater()
        self.harness._wall_verification_instruction_timer.deleteLater()
        _qt_application.processEvents()

    def test_same_context_is_current_and_level_or_image_changes_are_stale(self) -> None:
        state = self.harness.snapshot()
        self.assertTrue(self.harness._wall_verification_context_is_current(state))
        self.harness.current_level.image_path = "changed-plan.png"
        self.assertFalse(self.harness._wall_verification_context_is_current(state))
        self.harness.current_level.image_path = None
        self.harness.current_level.index += 1
        self.assertFalse(self.harness._wall_verification_context_is_current(state))

    def test_wall_opening_stair_and_transform_changes_invalidate_snapshot(self) -> None:
        changes = (
            lambda: self.harness.current_level.vertex_data.add_vertex(10.0, 20.0),
            lambda: self.harness.current_level.doorways.append(
                DoorwayData(10.0, 20.0, 1.0, 2.0)
            ),
            lambda: self.harness.current_level.open_spaces.append(
                OpenSpaceData("hole", 1.0, 1.0, 10.0, 10.0)
            ),
            lambda: self.harness.stairs.append("updated stair"),
            lambda: setattr(self.harness.current_level, "scale", 2.0),
            lambda: setattr(self.harness.current_level, "offset_x_meters", 2.0),
            lambda: setattr(self.harness.current_level, "plan_meters_per_pixel", 0.03),
        )
        for change in changes:
            with self.subTest(change=change):
                state = self.harness.snapshot()
                change()
                self.assertFalse(
                    self.harness._wall_verification_context_is_current(state)
                )

    def test_stale_pending_or_preview_is_dismissed_but_current_state_is_kept(self) -> None:
        state = self.harness.snapshot()
        self.harness._wall_verification_preview = state
        self.harness._invalidate_stale_wall_verification()
        self.assertEqual(self.harness.dismissed_messages, [])
        self.harness.current_level.scale = 2.0
        self.harness._invalidate_stale_wall_verification()
        self.assertIn(
            "plan, walls, or openings changed", self.harness.dismissed_messages[-1]
        )
        self.harness._wall_verification_preview = None
        self.harness._wall_verification_active_job_id = "review"
        self.harness._wall_verification_runtimes["review"] = state
        self.harness._invalidate_stale_wall_verification()
        self.assertEqual(len(self.harness.dismissed_messages), 2)

    def test_doorways_and_open_spaces_are_immutable_protected_polygons(self) -> None:
        self.harness.current_level.doorways = [DoorwayData(10.0, 20.0, 1.0, 2.0)]
        doorway_corners = [(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)]
        self.harness.canvas = SimpleNamespace(
            _get_doorway_corners=lambda _doorway: doorway_corners
        )
        self.harness.current_level.open_spaces = [
            OpenSpaceData("hole", 5.0, 6.0, 15.0, 16.0)
        ]

        polygons = self.harness._wall_verification_excluded_polygons()
        doorway_corners.clear()

        self.assertIsInstance(polygons, tuple)
        self.assertEqual(
            polygons[0], ((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0))
        )
        self.assertEqual(
            polygons[1], ((5.0, 6.0), (15.0, 6.0), (15.0, 16.0), (5.0, 16.0))
        )

    def test_window_span_protection_is_in_image_coordinates_for_current_level_only(self) -> None:
        self.harness.current_level.scale = 1.7
        self.harness.current_level.offset_x_meters = 3.0
        target = _window_target(self.harness.current_level.index)
        self.harness._canvas_opening_targets_by_key = {
            target.key: target,
            "other": _window_target(self.harness.current_level.index + 1),
        }

        polygons = self.harness._wall_verification_excluded_polygons()
        start = level_world_to_image_xy(self.harness.current_level, 1.0, 0.0)
        end = level_world_to_image_xy(self.harness.current_level, 3.0, 0.0)

        self.assertEqual(len(polygons), 1)
        self.assertAlmostEqual(polygons[0][0][0], start[0])
        self.assertAlmostEqual(polygons[0][1][0], end[0])
        self.assertAlmostEqual(abs(polygons[0][0][1] - start[1]), 2.0)
        self.assertAlmostEqual(abs(polygons[0][3][1] - start[1]), 2.0)

    def test_empty_canvas_shows_initial_instructions_with_either_setting(self) -> None:
        label = self.harness.wall_verification_instructions_label
        timer = self.harness._wall_verification_instruction_timer

        for enabled in (False, True):
            with self.subTest(verification_enabled=enabled):
                self.harness._generation_settings.include_llm_wall_verification = enabled
                self.harness._update_wall_verification_instructions()
                self.assertEqual(
                    label.text(),
                    "Click 'Generate walls', adjust the green preview, then click "
                    "'Confirm walls'",
                )
                self.assertFalse(label.isHidden())
                self.assertTrue(timer.isActive())

    def test_existing_vertices_hide_idle_instructions_with_either_setting(self) -> None:
        self.harness.current_level.vertex_data.add_vertex(10.0, 20.0)

        for enabled in (False, True):
            with self.subTest(verification_enabled=enabled):
                self.harness._generation_settings.include_llm_wall_verification = enabled
                self.harness._update_wall_verification_instructions()
                self.assertTrue(self.harness.wall_verification_instructions_label.isHidden())
                self.assertFalse(self.harness._wall_verification_instruction_timer.isActive())

    def test_enabled_review_stages_override_idle_state_with_existing_vertices(self) -> None:
        self.harness.current_level.vertex_data.add_vertex(10.0, 20.0)
        self.harness._generation_settings.include_llm_wall_verification = True
        label = self.harness.wall_verification_instructions_label
        timer = self.harness._wall_verification_instruction_timer

        self.harness._wall_verification_active_job_id = "review"
        self.harness._update_wall_verification_instructions()
        self.assertEqual(label.text(), "Wait for LLM wall verification")
        self.assertFalse(label.isHidden())
        self.assertTrue(timer.isActive())

        self.harness._wall_verification_active_job_id = None
        self.harness._wall_verification_preview = self.harness.snapshot()
        self.harness._update_wall_verification_instructions()
        self.assertIn("Red vertices and lines", label.text())
        self.assertIn("Click a red line or point", label.text())
        self.assertIn("Accept LLM proposed changes", label.text())
        self.assertIn("Right-click", label.text())
        self.assertFalse(label.isHidden())
        self.assertTrue(timer.isActive())

        self.harness._wall_verification_preview = None
        self.harness._update_wall_verification_instructions()
        self.assertTrue(label.isHidden())
        self.assertFalse(timer.isActive())

    def test_disabled_verification_does_not_expose_stale_review_instructions(self) -> None:
        label = self.harness.wall_verification_instructions_label
        for stage in ("pending", "preview"):
            for has_vertices in (False, True):
                with self.subTest(stage=stage, has_vertices=has_vertices):
                    self.harness.current_level.vertex_data.vertices.clear()
                    if has_vertices:
                        self.harness.current_level.vertex_data.add_vertex(10.0, 20.0)
                    self.harness._wall_verification_active_job_id = (
                        "review" if stage == "pending" else None
                    )
                    self.harness._wall_verification_preview = (
                        self.harness.snapshot() if stage == "preview" else None
                    )
                    self.harness._update_wall_verification_instructions()
                    self.assertEqual(label.isHidden(), has_vertices)
                    self.assertEqual(
                        self.harness._wall_verification_instruction_timer.isActive(),
                        not has_vertices,
                    )
                    if not has_vertices:
                        self.assertEqual(
                            label.text(),
                            "Click 'Generate walls', adjust the green preview, then "
                            "click 'Confirm walls'",
                        )

    def test_deleting_last_vertex_restores_initial_instructions(self) -> None:
        graph = self.harness.current_level.vertex_data
        vertex = graph.add_vertex(10.0, 20.0)
        self.harness._update_wall_verification_instructions()
        self.assertTrue(self.harness.wall_verification_instructions_label.isHidden())

        graph.delete_vertex(vertex.id)
        self.harness._update_wall_verification_instructions()

        self.assertFalse(self.harness.wall_verification_instructions_label.isHidden())
        self.assertTrue(self.harness._wall_verification_instruction_timer.isActive())

    def test_shutdown_stops_instruction_animation(self) -> None:
        self.harness._update_wall_verification_instructions()
        timer = self.harness._wall_verification_instruction_timer
        self.assertTrue(timer.isActive())
        self.harness._is_shutdown = True
        self.harness._update_wall_verification_instructions()
        self.assertFalse(timer.isActive())

    def test_instruction_animation_cycles_white_and_red_without_scene_updates(self) -> None:
        self.harness._generation_settings.include_llm_wall_verification = True
        self.harness._update_wall_verification_instructions()
        label = self.harness.wall_verification_instructions_label
        self.harness._wall_verification_instruction_phase = 0
        self.harness._animate_wall_verification_instructions()
        self.assertEqual(
            label.palette().color(QPalette.ColorRole.WindowText).name(), "#ffffff"
        )
        self.harness._wall_verification_instruction_phase = 12
        self.harness._animate_wall_verification_instructions()
        self.assertEqual(
            label.palette().color(QPalette.ColorRole.WindowText).name(), "#ff0000"
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
