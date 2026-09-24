# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import copy
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from PIL import Image
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.automatic_wall_detection import (
    PlanWallAnalysis,
    PlanWallDetectionOptions,
    PlanWallDetectionResult,
)
from housemaker.generation_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
)
from housemaker.level_coordinates import level_image_to_world_xy
from housemaker.main import BlueprintWorkspace
from housemaker.models import VertexData

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test doubles ###
class _FakeSignal:
    """Minimal signal stand-in for a manually completed worker."""

    def __init__(self) -> None:
        self._slots: list[object] = []

    def connect(self, slot: object) -> None:
        self._slots.append(slot)

    def emit(self) -> None:
        for slot in tuple(self._slots):
            slot()  # type: ignore[operator]


class _FakeWallDetectionThread:
    """Controllable wall-analysis worker that never reads the plan image."""

    instances: ClassVar[list[_FakeWallDetectionThread]] = []

    def __init__(
        self,
        image_path: Path,
        parent: object | None = None,
    ) -> None:
        self.image_path = Path(image_path)
        self.parent = parent
        self.finished = _FakeSignal()
        self.analysis: PlanWallAnalysis | None = None
        self.error_message: str | None = None
        self.was_cancelled = False
        self.running = False
        self.interruption_requested = False
        self.deleted_later = False
        type(self).instances.append(self)

    def start(self) -> None:
        self.running = True

    def finish(self) -> None:
        self.running = False
        self.finished.emit()

    def isRunning(self) -> bool:
        return self.running

    def requestInterruption(self) -> None:
        self.interruption_requested = True

    def wait(self, _milliseconds: int) -> bool:
        self.running = False
        return True

    def deleteLater(self) -> None:
        self.deleted_later = True


# ### Fixture helpers ###
def _add_existing_wall(vertex_data: VertexData) -> None:
    first = vertex_data.add_vertex(5.0, 5.0)
    second = vertex_data.add_vertex(90.0, 5.0)
    vertex_data.add_edge(first.id, second.id)


# ### Main-workspace integration tests ###
class PlanWallGenerationMainTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeWallDetectionThread.instances.clear()
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_path = Path(self._temporary_directory.name)
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                self.temporary_path / "settings.json"
            )
        )
        self.workspace.resize(1200, 800)
        self.workspace.show()
        _qt_application.processEvents()

        self.reconstruction_calls: list[
            tuple[PlanWallAnalysis, PlanWallDetectionOptions, VertexData]
        ] = []
        self._reconstruction_patch = patch(
            "housemaker.main.reconstruct_plan_walls",
            side_effect=self._reconstruct_walls,
        )
        self.reconstruct_walls = self._reconstruction_patch.start()
        self.addCleanup(self._reconstruction_patch.stop)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _load_plan(self) -> Path:
        image_path = self.temporary_path / "corrected-plan.png"
        Image.new("1", (100, 80), 1).save(image_path)
        self.workspace._set_current_level_image(str(image_path))
        return image_path.resolve()

    def _start_detection(self) -> tuple[_FakeWallDetectionThread, str]:
        with patch(
            "housemaker.main._PlanWallDetectionThread",
            _FakeWallDetectionThread,
        ):
            self.workspace._handle_generate_walls_clicked()

        self.assertEqual(len(_FakeWallDetectionThread.instances), 1)
        thread = _FakeWallDetectionThread.instances[0]
        runtime = self.workspace._plan_wall_detection_runtimes[
            self.workspace.current_level.index
        ]
        return thread, runtime.job_id

    def _finish_successful_detection(
        self,
    ) -> tuple[_FakeWallDetectionThread, str, PlanWallAnalysis]:
        thread, job_id = self._start_detection()
        analysis = PlanWallAnalysis(
            image_width=100,
            image_height=80,
            line_evidence=(),
        )
        thread.analysis = analysis
        thread.finish()
        return thread, job_id, analysis

    def _reconstruct_walls(
        self,
        analysis: PlanWallAnalysis,
        options: PlanWallDetectionOptions,
        *,
        existing_vertex_data: VertexData,
    ) -> PlanWallDetectionResult:
        self.reconstruction_calls.append(
            (analysis, options, existing_vertex_data)
        )
        candidate = existing_vertex_data.clone()
        preview_y = float(options.maximum_wall_separation_pixels)
        first = candidate.add_vertex(10.0, preview_y)
        second = candidate.add_vertex(80.0, preview_y)
        candidate.add_edge(first.id, second.id)
        return PlanWallDetectionResult(
            vertex_data=candidate,
            added_edge_count=1,
        )

    def test_generate_button_follows_source_and_async_worker_state(self) -> None:
        self.assertEqual(self.workspace.generate_walls_button.text(), "Generate walls")
        self.assertFalse(self.workspace.generate_walls_button.isEnabled())
        self.assertFalse(self.workspace.plan_wall_controls_group.isEnabled())

        source_path = self._load_plan()

        self.assertTrue(self.workspace.generate_walls_button.isEnabled())
        thread, job_id = self._start_detection()
        self.assertEqual(thread.image_path, source_path)
        self.assertTrue(thread.running)
        self.assertFalse(self.workspace.generate_walls_button.isEnabled())
        self.assertEqual(
            self.workspace.generate_walls_button.text(),
            "Generating walls...",
        )
        self.assertFalse(self.workspace.plan_wall_controls_group.isEnabled())

        thread.analysis = PlanWallAnalysis(100, 80, ())
        thread.finish()

        self.assertTrue(thread.deleted_later)
        self.assertTrue(self.workspace.generate_walls_button.isEnabled())
        self.assertEqual(
            self.workspace.generate_walls_button.text(),
            "Confirm walls",
        )
        self.assertTrue(self.workspace.plan_wall_controls_group.isEnabled())
        completed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(
            completed_job.status,  # type: ignore[union-attr]
            JOB_STATUS_COMPLETED,
        )

    def test_slider_rebuilds_preview_from_cached_analysis(self) -> None:
        self._load_plan()
        _thread, _job_id, analysis = self._finish_successful_detection()
        self.assertEqual(len(self.reconstruction_calls), 1)
        first_preview = self.workspace.canvas.get_generated_wall_preview()
        self.assertIsNotNone(first_preview)
        assert first_preview is not None
        initial_y = first_preview.vertices[-1].y

        new_maximum = self.workspace.maximum_wall_separation_slider.value() + 7
        self.workspace.maximum_wall_separation_slider.setValue(new_maximum)

        self.assertEqual(
            self.workspace.generate_walls_button.text(),
            "Confirm walls",
        )
        self.assertFalse(self.workspace.generate_walls_button.isEnabled())
        self.assertTrue(self.workspace._plan_wall_preview_refresh_timer.isActive())
        self.assertTrue(self.workspace._refresh_plan_wall_preview())

        self.assertEqual(len(self.reconstruction_calls), 2)
        self.assertIs(self.reconstruction_calls[0][0], analysis)
        self.assertIs(self.reconstruction_calls[1][0], analysis)
        self.assertEqual(
            self.reconstruction_calls[1][1].maximum_wall_separation_pixels,
            float(new_maximum),
        )
        second_preview = self.workspace.canvas.get_generated_wall_preview()
        self.assertIsNotNone(second_preview)
        assert second_preview is not None
        self.assertNotEqual(second_preview.vertices[-1].y, initial_y)
        self.assertEqual(second_preview.vertices[-1].y, float(new_maximum))
        self.assertEqual(len(_FakeWallDetectionThread.instances), 1)
        self.assertTrue(self.workspace.generate_walls_button.isEnabled())

    def test_maximum_vertex_distance_is_a_live_explained_preview_control(
        self,
    ) -> None:
        labels = {
            label.text()
            for label in self.workspace.plan_wall_controls_group.findChildren(QLabel)
        }
        self.assertIn("Maximum vertex distance", labels)
        self.assertEqual(
            self.workspace.maximum_vertex_distance_slider.value(),
            round(PlanWallDetectionOptions().maximum_vertex_distance_pixels),
        )
        tooltip = self.workspace.maximum_vertex_distance_slider.toolTip().lower()
        self.assertIn("generated", tooltip)
        self.assertIn("closer", tooltip)
        self.assertTrue("merge" in tooltip or "consolidat" in tooltip)

        self._load_plan()
        _thread, _job_id, analysis = self._finish_successful_detection()
        new_distance = self.workspace.maximum_vertex_distance_slider.value() + 3
        self.workspace.maximum_vertex_distance_slider.setValue(new_distance)

        self.assertTrue(self.workspace._plan_wall_preview_refresh_timer.isActive())
        self.assertTrue(self.workspace._refresh_plan_wall_preview())
        self.assertEqual(len(self.reconstruction_calls), 2)
        self.assertIs(self.reconstruction_calls[1][0], analysis)
        self.assertEqual(
            self.reconstruction_calls[1][1].maximum_vertex_distance_pixels,
            float(new_distance),
        )
        self.assertEqual(
            self.workspace.maximum_vertex_distance_value_label.text(),
            f"{new_distance} px",
        )

    def test_confirm_commits_once_and_shared_undo_restores_baseline(self) -> None:
        self._load_plan()
        _add_existing_wall(self.workspace.current_level.vertex_data)
        self.workspace.current_level.scale = 1.75
        self.workspace.current_level.offset_x_meters = 2.5
        self.workspace.current_level.offset_y_meters = -3.0
        baseline = self.workspace.current_level.vertex_data.to_dict()
        baseline_offsets = (
            self.workspace.current_level.offset_x_meters,
            self.workspace.current_level.offset_y_meters,
        )
        anchored_world_position = level_image_to_world_xy(
            self.workspace.current_level,
            5.0,
            5.0,
        )
        graph_identity = id(self.workspace.current_level.vertex_data)
        self._finish_successful_detection()
        geometry_events: list[str] = []
        self.workspace.canvas.geometry_changed.connect(
            lambda: geometry_events.append("changed")
        )
        undo_count = len(self.workspace._canvas_undo_stack)

        self.workspace._handle_generate_walls_clicked()

        committed = self.workspace.current_level.vertex_data
        self.assertEqual(id(committed), graph_identity)
        self.assertEqual(len(committed.edges), 2)
        self.assertEqual(
            level_image_to_world_xy(self.workspace.current_level, 5.0, 5.0),
            anchored_world_position,
        )
        self.assertEqual(geometry_events, ["changed"])
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count + 1)
        self.assertIsNone(self.workspace.canvas.get_generated_wall_preview())
        self.assertEqual(
            self.workspace.generate_walls_button.text(),
            "Generate walls",
        )
        self.assertFalse(self.workspace.plan_wall_controls_group.isEnabled())

        self.workspace._handle_canvas_undo_requested()

        self.assertEqual(id(self.workspace.current_level.vertex_data), graph_identity)
        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), baseline)
        self.assertEqual(
            (
                self.workspace.current_level.offset_x_meters,
                self.workspace.current_level.offset_y_meters,
            ),
            baseline_offsets,
        )
        self.assertEqual(len(self.workspace._canvas_undo_stack), undo_count)

    def test_stale_topology_rejects_finished_analysis(self) -> None:
        self._load_plan()
        thread, job_id = self._start_detection()
        self.workspace.current_level.vertex_data.add_vertex(25.0, 25.0)
        thread.analysis = PlanWallAnalysis(100, 80, ())

        with patch("housemaker.main.QMessageBox.warning") as warning:
            thread.finish()

        self.assertEqual(self.reconstruction_calls, [])
        self.assertIsNone(self.workspace.canvas.get_generated_wall_preview())
        self.assertIsNone(self.workspace._plan_wall_preview_session)
        self.assertEqual(
            self.workspace.generate_walls_button.text(),
            "Generate walls",
        )
        self.assertTrue(self.workspace.generate_walls_button.isEnabled())
        warning.assert_called_once()
        self.assertIn("changed during analysis", warning.call_args.args[2])
        failed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(
            failed_job.status,  # type: ignore[union-attr]
            JOB_STATUS_FAILED,
        )

    def test_stopped_worker_can_still_be_cancelled_before_finished_slot(
        self,
    ) -> None:
        self._load_plan()
        thread, job_id = self._start_detection()
        thread.running = False
        thread.analysis = PlanWallAnalysis(100, 80, ())

        self.assertTrue(self.workspace.job_manager.cancel_job(job_id))
        runtime = self.workspace._plan_wall_detection_runtimes[
            self.workspace.current_level.index
        ]
        self.assertTrue(runtime.cancel_requested)

        thread.finished.emit()

        job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(job.status, JOB_STATUS_CANCELLED)  # type: ignore[union-attr]
        self.assertIsNone(self.workspace._plan_wall_preview_session)
        self.assertIsNone(self.workspace.canvas.get_generated_wall_preview())

    def test_confirm_commits_pending_transform_before_pivot_compensation(
        self,
    ) -> None:
        self._load_plan()
        _add_existing_wall(self.workspace.current_level.vertex_data)
        self._finish_successful_detection()
        level = self.workspace.current_level

        self.workspace._handle_level_scale_changed(1.5)
        self.workspace._handle_level_x_offset_changed(1.2)
        pending = self.workspace._pending_level_transform
        self.assertIsNotNone(pending)
        assert pending is not None
        expected_level = copy.copy(level)
        expected_level.scale = pending.scale
        expected_level.offset_x_meters = pending.offset_x_meters
        expected_level.offset_y_meters = pending.offset_y_meters
        expected_anchor = level_image_to_world_xy(expected_level, 5.0, 5.0)

        self.workspace._handle_generate_walls_clicked()

        self.assertIsNone(self.workspace._pending_level_transform)
        self.assertAlmostEqual(level.scale, 1.5)
        actual_anchor = level_image_to_world_xy(level, 5.0, 5.0)
        self.assertAlmostEqual(actual_anchor[0], expected_anchor[0])
        self.assertAlmostEqual(actual_anchor[1], expected_anchor[1])
        self.assertEqual(len(self.workspace._canvas_undo_stack), 2)

        self.workspace._handle_canvas_undo_requested()
        self.assertEqual(len(level.vertex_data.edges), 1)
        self.assertAlmostEqual(level.scale, 1.5)
        self.assertAlmostEqual(level.offset_x_meters, 1.2)

        self.workspace._handle_canvas_undo_requested()
        self.assertAlmostEqual(level.scale, 1.0)
        self.assertAlmostEqual(level.offset_x_meters, 0.0)

    def test_confirm_preserves_scaled_isolated_vertex_world_position(self) -> None:
        self._load_plan()
        level = self.workspace.current_level
        level.vertex_data.add_vertex(5.0, 5.0)
        level.scale = 1.75
        level.offset_x_meters = 2.5
        level.offset_y_meters = -3.0
        expected_position = level_image_to_world_xy(level, 5.0, 5.0)
        self._finish_successful_detection()

        self.workspace._handle_generate_walls_clicked()

        self.assertEqual(
            level_image_to_world_xy(level, 5.0, 5.0),
            expected_position,
        )

    def test_malformed_candidate_cannot_replace_existing_graph(self) -> None:
        self._load_plan()
        _add_existing_wall(self.workspace.current_level.vertex_data)
        baseline = self.workspace.current_level.vertex_data.to_dict()

        def build_malformed_result(
            _analysis: PlanWallAnalysis,
            _options: PlanWallDetectionOptions,
            *,
            existing_vertex_data: VertexData,
        ) -> PlanWallDetectionResult:
            del existing_vertex_data
            replacement = VertexData()
            first = replacement.add_vertex(20.0, 20.0)
            second = replacement.add_vertex(70.0, 20.0)
            replacement.add_edge(first.id, second.id)
            return PlanWallDetectionResult(replacement, 1)

        self.reconstruct_walls.side_effect = build_malformed_result
        _thread, job_id, _analysis = self._finish_successful_detection()

        self.assertEqual(self.workspace.current_level.vertex_data.to_dict(), baseline)
        self.assertIsNone(self.workspace.canvas.get_generated_wall_preview())
        self.assertFalse(self.workspace.generate_walls_button.isEnabled())
        self.assertIn(
            "cannot replace existing",
            self.workspace.plan_wall_generation_status_label.text(),
        )
        job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(job.status, JOB_STATUS_FAILED)  # type: ignore[union-attr]

    def test_start_refreshes_an_externally_replaced_plan(self) -> None:
        image_path = self._load_plan()
        Image.new("1", (130, 90), 0).save(image_path)

        thread, _job_id = self._start_detection()

        self.assertEqual(thread.image_path, image_path)
        self.assertEqual(self.workspace.canvas.get_image_size_pixels(), (130.0, 90.0))
        self.assertEqual(
            self.workspace.current_level.image_size_pixels,
            (130.0, 90.0),
        )
        self.assertEqual(
            self.workspace.canvas.get_blueprint_image_revision(),
            self.workspace._plan_wall_detection_runtimes[
                self.workspace.current_level.index
            ].source_revision,
        )

    def test_confirm_runs_one_scheduled_3d_rebuild(self) -> None:
        self._load_plan()
        _add_existing_wall(self.workspace.current_level.vertex_data)
        self._finish_successful_detection()
        self.workspace.workspace_tabs.setCurrentWidget(
            self.workspace.scene_3d_workspace
        )
        QTest.qWait(30)
        _qt_application.processEvents()

        with patch.object(
            self.workspace,
            "_refresh_viewer_preview",
            wraps=self.workspace._refresh_viewer_preview,
        ) as refresh:
            self.workspace._handle_generate_walls_clicked()
            QTest.qWait(30)
            _qt_application.processEvents()

        self.assertEqual(refresh.call_count, 1)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
