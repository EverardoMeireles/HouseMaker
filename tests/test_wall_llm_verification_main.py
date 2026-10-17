# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import test_plan_wall_generation_main as wall_fixtures
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from housemaker.generation_jobs import JOB_STATUS_CANCELLED, JOB_STATUS_FAILED
from housemaker.models import OpenSpaceData, Vertex, VertexData
from housemaker.wall_llm_verification import (
    WallAdditionProposal,
    WallVerificationError,
    WallVerificationResult,
)


# ### Test doubles ###
class _FakeVerificationThread(wall_fixtures._FakeWallDetectionThread):
    """A controllable semantic review that never calls a paid provider."""

    instances: ClassVar[list[_FakeVerificationThread]] = []

    def __init__(
        self,
        image_path: Path,
        vertex_data: VertexData,
        *,
        api_key: str,
        model: str,
        excluded_polygons: tuple,
        parent: object | None = None,
    ) -> None:
        super().__init__(image_path, parent)
        self.vertex_data = vertex_data.clone()
        self.api_key = api_key
        self.model = model
        self.excluded_polygons = excluded_polygons
        self.result: WallVerificationResult | None = None


# ### Main-workspace integration tests ###
class WallVerificationMainTests(unittest.TestCase):
    # Reuse the deterministic fixture without inheriting its test cases.
    tearDown = wall_fixtures.PlanWallGenerationMainTests.tearDown
    _load_plan = wall_fixtures.PlanWallGenerationMainTests._load_plan
    _start_detection = wall_fixtures.PlanWallGenerationMainTests._start_detection
    _finish_successful_detection = (
        wall_fixtures.PlanWallGenerationMainTests._finish_successful_detection
    )
    _reconstruct_walls = wall_fixtures.PlanWallGenerationMainTests._reconstruct_walls
    _reconstruct_unchanged_walls = (
        wall_fixtures.PlanWallGenerationMainTests._reconstruct_unchanged_walls
    )

    def setUp(self) -> None:
        _FakeVerificationThread.instances.clear()
        wall_fixtures.PlanWallGenerationMainTests.setUp(self)
        verification_patch = patch(
            "housemaker.main._WallVerificationThread",
            _FakeVerificationThread,
        )
        verification_patch.start()
        self.addCleanup(verification_patch.stop)
        self.workspace.settings_widget.openai_api_key_edit.setText(
            "unit-test-not-a-real-api-key"
        )

    def _confirm_with_review(self) -> _FakeVerificationThread:
        self.workspace.settings_widget.include_llm_wall_verification_checkbox.setChecked(
            True
        )
        self._load_plan()
        self._finish_successful_detection()
        self.workspace._handle_generate_walls_clicked()
        self.assertEqual(len(_FakeVerificationThread.instances), 1)
        return _FakeVerificationThread.instances[0]

    def _result(self, thread: _FakeVerificationThread) -> WallVerificationResult:
        first, second = thread.vertex_data.vertices[-2:]
        next_id = max(vertex.id for vertex in thread.vertex_data.vertices) + 1
        return WallVerificationResult(
            proposals=(
                WallAdditionProposal(
                    "left-addition",
                    first,
                    Vertex(next_id, first.x, first.y - 20.0),
                    0.95,
                    "A visible wall continues vertically from the left endpoint.",
                ),
                WallAdditionProposal(
                    "right-addition",
                    second,
                    Vertex(next_id + 1, second.x, second.y - 20.0),
                    0.95,
                    "A visible wall continues vertically from the right endpoint.",
                ),
            ),
            summary="Two missing wall continuations.",
        )

    def _finish_review(self, thread: _FakeVerificationThread) -> None:
        thread.result = self._result(thread)
        thread.finish()

    def test_default_does_not_launch_verification(self) -> None:
        self._load_plan()
        self._finish_successful_detection()
        self.workspace._handle_generate_walls_clicked()

        self.assertEqual(_FakeVerificationThread.instances, [])
        self.assertIsNone(self.workspace._wall_verification_active_job_id)
        self.assertEqual(len(self.workspace.current_level.vertex_data.edges), 1)
        self.assertEqual(self.workspace.generate_walls_button.text(), "Generate walls")
        self.assertTrue(self.workspace.wall_verification_instructions_label.isHidden())
        self.assertFalse(self.workspace._wall_verification_instruction_timer.isActive())

    def test_first_and_last_vertex_changes_refresh_instructions_before_mesh_delay(
        self,
    ) -> None:
        label = self.workspace.wall_verification_instructions_label
        timer = self.workspace._wall_verification_instruction_timer
        self.assertFalse(label.isHidden())
        self.assertTrue(timer.isActive())
        vertex = self.workspace.current_level.vertex_data.add_vertex(10.0, 20.0)
        self.workspace._pending_wall_vertex_mesh_update = True

        self.workspace.canvas.geometry_changed.emit()

        self.assertTrue(label.isHidden())
        self.assertFalse(timer.isActive())
        self.workspace.current_level.vertex_data.delete_vertex(vertex.id)

        self.workspace.canvas.geometry_changed.emit()

        self.assertFalse(label.isHidden())
        self.assertEqual(
            label.text(),
            "Click 'Generate walls', adjust the green preview, then click 'Confirm walls'",
        )
        self.assertTrue(timer.isActive())

    def test_enabled_review_launches_only_after_confirmation_with_selected_model(
        self,
    ) -> None:
        settings = self.workspace.settings_widget
        settings.include_llm_wall_verification_checkbox.setChecked(True)
        settings.wall_verification_model_combo.setCurrentIndex(
            settings.wall_verification_model_combo.findData("gpt-5.6-terra")
        )
        self._load_plan()
        self._finish_successful_detection()
        self.assertEqual(_FakeVerificationThread.instances, [])
        self.assertEqual(len(self.workspace.current_level.vertex_data.edges), 0)

        self.workspace._handle_generate_walls_clicked()

        thread = _FakeVerificationThread.instances[0]
        self.assertEqual(thread.model, "gpt-5.6-terra")
        self.assertEqual(thread.image_path, Path(self.workspace.current_level.image_path))
        self.assertEqual(
            thread.vertex_data.to_dict(),
            self.workspace.current_level.vertex_data.to_dict(),
        )
        self.assertTrue(thread.running)
        self.assertFalse(self.workspace.generate_walls_button.isEnabled())
        self.assertIn("Verif", self.workspace.generate_walls_button.text())
        self.assertTrue(self.workspace._wall_verification_instruction_timer.isActive())
        self.assertEqual(
            self.workspace.wall_verification_instructions_label.text(),
            "Wait for LLM wall verification",
        )

    def test_unchanged_preview_can_start_verification_without_rebuilding_walls(
        self,
    ) -> None:
        settings = self.workspace.settings_widget
        settings.include_llm_wall_verification_checkbox.setChecked(True)
        settings.wall_verification_model_combo.setCurrentIndex(
            settings.wall_verification_model_combo.findData("gpt-5.6-terra")
        )
        self._load_plan()
        wall_fixtures._add_existing_wall(self.workspace.current_level.vertex_data)
        graph = self.workspace.current_level.vertex_data
        baseline = graph.to_dict()
        revision = graph.revision
        undo_count = len(self.workspace._canvas_undo_stack)
        self.reconstruct_walls.side_effect = self._reconstruct_unchanged_walls
        self._finish_successful_detection()
        self.assertEqual(_FakeVerificationThread.instances, [])
        self.assertTrue(self.workspace.generate_walls_button.isEnabled())

        with (
            patch.object(self.workspace.canvas, "commit_generated_wall_preview") as commit,
            patch.object(self.workspace, "_schedule_viewer_preview_refresh") as rebuild,
        ):
            self.workspace._handle_generate_walls_clicked()

        commit.assert_not_called()
        rebuild.assert_not_called()
        self.assertEqual(len(_FakeVerificationThread.instances), 1)
        thread = _FakeVerificationThread.instances[0]
        self.assertEqual(thread.model, "gpt-5.6-terra")
        self.assertTrue(thread.running)
        self.assertEqual(thread.vertex_data.to_dict(), baseline)
        self.assertEqual(graph.to_dict(), baseline)
        self.assertEqual(graph.revision, revision)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)
        self.assertIsNone(self.workspace._plan_wall_preview_session)
        self.assertIsNone(self.workspace.canvas.get_generated_wall_preview())
        self.assertEqual(
            self.workspace.wall_verification_instructions_label.text(),
            "Wait for LLM wall verification",
        )

    def test_empty_wall_preview_can_start_verification(self) -> None:
        settings = self.workspace.settings_widget
        settings.include_llm_wall_verification_checkbox.setChecked(True)
        self._load_plan()
        graph = self.workspace.current_level.vertex_data
        baseline = graph.to_dict()
        revision = graph.revision
        undo_count = len(self.workspace._canvas_undo_stack)
        self.reconstruct_walls.side_effect = self._reconstruct_unchanged_walls
        self._finish_successful_detection()

        self.assertTrue(self.workspace.generate_walls_button.isEnabled())
        self.workspace._handle_generate_walls_clicked()

        self.assertEqual(len(_FakeVerificationThread.instances), 1)
        thread = _FakeVerificationThread.instances[0]
        self.assertTrue(thread.running)
        self.assertEqual(thread.vertex_data.to_dict(), baseline)
        self.assertEqual(graph.to_dict(), baseline)
        self.assertEqual(graph.revision, revision)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)
        self.assertIsNone(self.workspace._plan_wall_preview_session)

    def test_finished_review_is_only_an_overlay_until_acceptance(self) -> None:
        thread = self._confirm_with_review()
        confirmed = self.workspace.current_level.vertex_data.to_dict()
        undo_count = len(self.workspace._canvas_undo_stack)
        geometry_events: list[bool] = []
        self.workspace.canvas.geometry_changed.connect(
            lambda: geometry_events.append(True)
        )

        self._finish_review(thread)

        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)
        self.assertEqual(geometry_events, [])
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)
        self.assertIsNotNone(self.workspace._wall_verification_preview)
        self.assertEqual(
            self.workspace.canvas.selected_wall_verification_proposal_ids(),
            {"left-addition", "right-addition"},
        )
        self.assertEqual(
            self.workspace.generate_walls_button.text(),
            "Accept LLM proposed changes",
        )
        self.assertTrue(self.workspace.generate_walls_button.isEnabled())
        self.assertIn(
            "Accept LLM proposed changes",
            self.workspace.wall_verification_instructions_label.text(),
        )
        self.assertFalse(self.workspace.wall_verification_instructions_label.isHidden())
        self.assertTrue(self.workspace._wall_verification_instruction_timer.isActive())

    def test_accept_selected_additions_is_one_undo_transaction(self) -> None:
        thread = self._confirm_with_review()
        confirmed = self.workspace.current_level.vertex_data.to_dict()
        self._finish_review(thread)
        undo_count = len(self.workspace._canvas_undo_stack)
        first = thread.result.proposals[0]
        midpoint = self.workspace.canvas._image_to_widget(
            (first.start.x + first.end.x) / 2.0,
            (first.start.y + first.end.y) / 2.0,
        )
        QTest.mouseClick(
            self.workspace.canvas,
            Qt.MouseButton.LeftButton,
            pos=midpoint.toPoint(),
        )
        self.assertEqual(
            self.workspace.canvas.selected_wall_verification_proposal_ids(),
            {"right-addition"},
        )

        self.workspace._handle_generate_walls_clicked()

        graph = self.workspace.current_level.vertex_data
        self.assertEqual(len(graph.edges), len(confirmed["edges"]) + 1)
        self.assertEqual(len(graph.vertices), len(confirmed["vertices"]) + 1)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count + 1)
        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertEqual(self.workspace.canvas.selected_wall_verification_proposal_ids(), set())
        self.workspace._handle_canvas_undo_requested()
        self.assertEqual(graph.to_dict(), confirmed)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)

    def test_right_click_dismisses_proposals_but_retains_confirmed_walls(self) -> None:
        thread = self._confirm_with_review()
        confirmed = self.workspace.current_level.vertex_data.to_dict()
        self._finish_review(thread)
        undo_count = len(self.workspace._canvas_undo_stack)

        QTest.mouseClick(self.workspace.canvas, Qt.MouseButton.RightButton)

        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)
        self.assertEqual(self.workspace.generate_walls_button.text(), "Generate walls")
        self.assertTrue(self.workspace.wall_verification_instructions_label.isHidden())
        self.assertFalse(self.workspace._wall_verification_instruction_timer.isActive())

    def test_deselecting_all_proposals_disables_acceptance(self) -> None:
        thread = self._confirm_with_review()
        self._finish_review(thread)
        confirmed = self.workspace.current_level.vertex_data.to_dict()
        undo_count = len(self.workspace._canvas_undo_stack)
        for proposal in thread.result.proposals:
            midpoint = self.workspace.canvas._image_to_widget(
                (proposal.start.x + proposal.end.x) / 2.0,
                (proposal.start.y + proposal.end.y) / 2.0,
            )
            QTest.mouseClick(
                self.workspace.canvas,
                Qt.MouseButton.LeftButton,
                pos=midpoint.toPoint(),
            )

        self.assertEqual(
            self.workspace.canvas.selected_wall_verification_proposal_ids(),
            set(),
        )
        self.assertFalse(self.workspace.generate_walls_button.isEnabled())
        self.workspace._accept_wall_verification_proposals()
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)

    def test_empty_review_retains_confirmed_walls_without_acceptance_mode(self) -> None:
        thread = self._confirm_with_review()
        confirmed = self.workspace.current_level.vertex_data.to_dict()
        thread.result = WallVerificationResult((), summary="No missing walls found.")

        thread.finish()

        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)
        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertIsNone(self.workspace._wall_verification_active_job_id)
        self.assertEqual(self.workspace.generate_walls_button.text(), "Generate walls")
        self.assertTrue(self.workspace.wall_verification_instructions_label.isHidden())
        self.assertFalse(self.workspace._wall_verification_instruction_timer.isActive())

    def test_right_click_cancels_pending_review_and_ignores_late_result(self) -> None:
        thread = self._confirm_with_review()
        job_id = self.workspace._wall_verification_active_job_id
        confirmed = self.workspace.current_level.vertex_data.to_dict()

        QTest.mouseClick(self.workspace.canvas, Qt.MouseButton.RightButton)

        self.assertTrue(thread.interruption_requested)
        self.assertIsNone(self.workspace._wall_verification_active_job_id)
        self._finish_review(thread)
        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)
        self.assertEqual(
            self.workspace.job_manager.get_job(job_id).status,
            JOB_STATUS_CANCELLED,
        )

    def test_jobs_cancel_callback_cancels_only_the_optional_review(self) -> None:
        thread = self._confirm_with_review()
        job_id = self.workspace._wall_verification_active_job_id
        confirmed = self.workspace.current_level.vertex_data.to_dict()

        self.assertTrue(self.workspace.job_manager.cancel_job(job_id))

        self.assertTrue(thread.interruption_requested)
        self.assertIsNone(self.workspace._wall_verification_active_job_id)
        self._finish_review(thread)
        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)
        self.assertEqual(
            self.workspace.job_manager.get_job(job_id).status,
            JOB_STATUS_CANCELLED,
        )

    def test_changed_graph_rejects_stale_response(self) -> None:
        thread = self._confirm_with_review()
        self.workspace.current_level.vertex_data.add_vertex(35.0, 15.0)
        updated = self.workspace.current_level.vertex_data.to_dict()

        with patch("housemaker.main.QMessageBox.warning"):
            self._finish_review(thread)

        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertIsNone(self.workspace._wall_verification_active_job_id)
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), updated)

    def test_cancelled_worker_finishing_does_not_replace_a_newer_review(self) -> None:
        first_thread = self._confirm_with_review()
        self.workspace._dismiss_wall_verification()
        self.workspace._start_wall_verification()
        next_job_id = self.workspace._wall_verification_active_job_id
        second_thread = _FakeVerificationThread.instances[-1]
        self.assertIsNot(first_thread, second_thread)

        self._finish_review(first_thread)

        self.assertEqual(
            self.workspace._wall_verification_active_job_id,
            next_job_id,
        )
        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertFalse(self.workspace.generate_walls_button.isEnabled())
        self._finish_review(second_thread)
        self.assertIsNotNone(self.workspace._wall_verification_preview)

    def test_changed_opening_metadata_rejects_stale_response(self) -> None:
        thread = self._confirm_with_review()
        confirmed = self.workspace.current_level.vertex_data.to_dict()
        self.workspace.current_level.open_spaces.append(
            OpenSpaceData("new-opening", 15.0, 15.0, 30.0, 30.0)
        )

        self._finish_review(thread)

        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertIsNone(self.workspace._wall_verification_active_job_id)
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)

    def test_acceptance_validation_error_preserves_walls_and_shows_reason(self) -> None:
        thread = self._confirm_with_review()
        self._finish_review(thread)
        confirmed = self.workspace.current_level.vertex_data.to_dict()
        undo_count = len(self.workspace._canvas_undo_stack)
        with (
            patch(
                "housemaker.wall_llm_verification.apply_wall_proposals",
                side_effect=WallVerificationError("The wall addition crosses a wall."),
            ),
            patch("housemaker.main.QMessageBox.critical") as error_dialog,
        ):
            self.workspace._accept_wall_verification_proposals()

        error_dialog.assert_called_once()
        self.assertIn("crosses a wall", error_dialog.call_args.args[2])
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)
        self.assertIsNotNone(self.workspace._wall_verification_preview)

    def test_changed_plan_image_rejects_stale_response(self) -> None:
        thread = self._confirm_with_review()
        confirmed = self.workspace.current_level.vertex_data.to_dict()
        Image.new("1", (110, 85), 1).save(thread.image_path)

        with patch("housemaker.main.QMessageBox.warning"):
            self._finish_review(thread)

        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)

    def test_worker_failure_shows_reason_without_undoing_confirmed_walls(self) -> None:
        thread = self._confirm_with_review()
        job_id = self.workspace._wall_verification_active_job_id
        confirmed = self.workspace.current_level.vertex_data.to_dict()
        thread.error_message = "The verification provider rejected the request."

        with patch("housemaker.main.QMessageBox.critical") as warning:
            thread.finish()

        warning.assert_called_once()
        self.assertIn(thread.error_message, warning.call_args.args[2])
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)
        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertEqual(
            self.workspace.job_manager.get_job(job_id).status,
            JOB_STATUS_FAILED,
        )

    def test_level_switch_cancels_review(self) -> None:
        thread = self._confirm_with_review()
        level = self.workspace.current_level
        confirmed = level.vertex_data.to_dict()

        self.workspace._handle_level_selection_changed(
            (self.workspace.current_level_index + 1) % len(self.workspace.levels)
        )

        self.assertTrue(thread.interruption_requested)
        self.assertIsNone(self.workspace._wall_verification_active_job_id)
        self._finish_review(thread)
        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertEqual(level.vertex_data.to_dict(), confirmed)

    def test_disabling_setting_cancels_review(self) -> None:
        thread = self._confirm_with_review()
        confirmed = self.workspace.current_level.vertex_data.to_dict()

        self.workspace.settings_widget.include_llm_wall_verification_checkbox.setChecked(
            False
        )

        self.assertTrue(thread.interruption_requested)
        self.assertIsNone(self.workspace._wall_verification_active_job_id)
        self._finish_review(thread)
        self.assertIsNone(self.workspace._wall_verification_preview)
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), confirmed)

    def test_shutdown_interrupts_and_retires_review_worker(self) -> None:
        thread = self._confirm_with_review()

        self.workspace.shutdown()

        self.assertTrue(thread.interruption_requested)
        self.assertFalse(thread.running)
        self.assertEqual(self.workspace._wall_verification_runtimes, {})
        self.assertIsNone(self.workspace._wall_verification_active_job_id)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
