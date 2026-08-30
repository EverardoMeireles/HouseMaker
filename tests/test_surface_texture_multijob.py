# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QThread
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from housemaker.generation_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    GenerationJobManager,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_texture_providers import SurfaceTextureResult
from housemaker.surface_texture_state import (
    SurfaceTextureData,
    SurfaceTextureVariant,
)
from housemaker.surface_texture_workspace import (
    SurfaceTextureWorker,
    SurfaceTextureGenerationWorkspace,
    SurfaceTextureRequest,
    _SavedSurfaceTextureOutput,
    _prepare_surface_texture_outputs,
)
from tests.test_surface_texture_workspace import (
    _colored_texture_png,
    _test_level,
    _texture_png,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)
_FIRST_WALL = "level:2/room:5/wall:1:2"
_SECOND_WALL = "level:2/room:5/wall:2:3"


# ### Test providers ###
class _ControlledSurfaceProvider:
    """Hold independent requests so tests can choose completion order."""

    def __init__(self) -> None:
        self.started = {
            _FIRST_WALL: threading.Event(),
            _SECOND_WALL: threading.Event(),
        }
        self.release = {
            _FIRST_WALL: threading.Event(),
            _SECOND_WALL: threading.Event(),
        }
        self.requests: list[SurfaceTextureRequest] = []
        self._request_lock = threading.Lock()

    def generate(self, request: SurfaceTextureRequest) -> SurfaceTextureResult:
        surface_id = request.surface_ids[0]
        with self._request_lock:
            self.requests.append(request)
        self.started[surface_id].set()
        if not self.release[surface_id].wait(timeout=10.0):
            raise RuntimeError("Controlled Surface provider timed out")
        color = (
            (190, 40, 30, 255)
            if surface_id == _FIRST_WALL
            else (20, 150, 210, 255)
        )
        return SurfaceTextureResult(
            provider="meshy",
            texture_png=_colored_texture_png(color),
            task_id=f"task-{surface_id.rsplit(':', maxsplit=1)[-1]}",
        )

    def release_all(self) -> None:
        for event in self.release.values():
            event.set()


class _LateOutputThread(QThread):
    """Inject a saved worker output while shutdown is waiting for the thread."""

    def __init__(self, inject_output) -> None:
        super().__init__()
        self._inject_output = inject_output
        self._is_running = True

    def isRunning(self) -> bool:  # type: ignore[override]
        return self._is_running

    def wait(self, time: int = 2**64 - 1) -> bool:  # type: ignore[override]
        self._inject_output()
        self._is_running = False
        return True


# ### Fixture helpers ###
def _request(surface_id: str, display_name: str = "") -> SurfaceTextureRequest:
    return SurfaceTextureRequest(
        provider="meshy",
        api_key="test-key",
        reference_pngs=(_texture_png(),),
        reference_frame_indices=(0,),
        surface_type="wall",
        surface_ids=(surface_id,),
        combined_area_m2=6.0,
        prompt=f"Texture {surface_id}",
        display_name=display_name,
    )


def _wait_until(predicate, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not predicate() and time.monotonic() < deadline:
        _qt_application.processEvents()
        QTest.qWait(5)
    if not predicate():
        raise AssertionError("Timed out waiting for the Surface job state")


# ### Multi-job tests ###
class SurfaceTextureMultiJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_path = Path(tempfile.mkdtemp())
        self.provider = _ControlledSurfaceProvider()
        self.manager = GenerationJobManager()
        self.workspace = SurfaceTextureGenerationWorkspace(
            provider=self.provider,
            asset_directory=self._temporary_path / "surface_assets",
            job_manager=self.manager,
        )
        self.workspace.set_levels([_test_level()])

    def tearDown(self) -> None:
        self.provider.release_all()
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        shutil.rmtree(self._temporary_path, ignore_errors=True)

    def test_disjoint_jobs_run_together_and_commit_in_reverse_order(self) -> None:
        self.assertTrue(self.workspace._start_generation(_request(_FIRST_WALL)))
        self.assertTrue(self.workspace._start_generation(_request(_SECOND_WALL)))
        self.assertTrue(self.provider.started[_FIRST_WALL].wait(timeout=2.0))
        self.assertTrue(self.provider.started[_SECOND_WALL].wait(timeout=2.0))
        self.assertEqual(len(self.workspace._generation_threads), 2)

        self.provider.release[_SECOND_WALL].set()
        _wait_until(
            lambda: any(
                assignment.surface_ids == (_SECOND_WALL,)
                for assignment in self.workspace.get_data().assignments
            )
        )
        self.assertTrue(self.workspace.is_generating)

        self.provider.release[_FIRST_WALL].set()
        _wait_until(lambda: not self.workspace.is_generating)

        assignments = self.workspace.get_data().assignments
        self.assertEqual(
            {assignment.surface_ids for assignment in assignments},
            {(_FIRST_WALL,), (_SECOND_WALL,)},
        )
        self.assertEqual(
            {job.status for job in self.manager.jobs()},
            {JOB_STATUS_COMPLETED},
        )
        self.assertEqual(
            len(list((self._temporary_path / "surface_assets").glob("*.png"))),
            6,
        )

    def test_same_target_is_rejected_without_starting_a_second_job(self) -> None:
        self.assertTrue(self.workspace._start_generation(_request(_FIRST_WALL)))
        self.assertTrue(self.provider.started[_FIRST_WALL].wait(timeout=2.0))

        self.assertFalse(
            self.workspace._start_generation(
                _request(_FIRST_WALL, "Duplicate request")
            )
        )

        self.assertEqual(len(self.workspace._generation_threads), 1)
        self.assertEqual(len(self.manager.jobs()), 1)
        self.assertIn("already using", self.workspace.status_label.text())
        self.provider.release[_FIRST_WALL].set()
        _wait_until(lambda: not self.workspace.is_generating)

    def test_cancel_one_job_discards_its_late_result_only(self) -> None:
        self.assertTrue(self.workspace._start_generation(_request(_FIRST_WALL)))
        self.assertTrue(self.workspace._start_generation(_request(_SECOND_WALL)))
        self.assertTrue(self.provider.started[_FIRST_WALL].wait(timeout=2.0))
        self.assertTrue(self.provider.started[_SECOND_WALL].wait(timeout=2.0))
        first_job_id = next(
            job_id
            for job_id, request in self.workspace._generation_requests.items()
            if request.surface_ids == (_FIRST_WALL,)
        )

        self.assertTrue(self.manager.cancel_job(first_job_id))
        _wait_until(
            lambda: (
                self.manager.get_job(first_job_id) is not None
                and self.manager.get_job(first_job_id).status
                == JOB_STATUS_CANCELLED
            )
        )
        self.provider.release[_SECOND_WALL].set()
        _wait_until(
            lambda: any(
                assignment.surface_ids == (_SECOND_WALL,)
                for assignment in self.workspace.get_data().assignments
            )
        )
        self.provider.release[_FIRST_WALL].set()
        _wait_until(lambda: not self.workspace.is_generating)

        self.assertEqual(
            [
                assignment.surface_ids
                for assignment in self.workspace.get_data().assignments
            ],
            [(_SECOND_WALL,)],
        )
        self.assertEqual(
            self.manager.get_job(first_job_id).status,
            JOB_STATUS_CANCELLED,
        )

    def test_late_cancel_is_finalized_when_worker_finished_was_already_queued(
        self,
    ) -> None:
        """A late GUI cancel must not leave the managed job cancelling forever."""

        managed_job = self.manager.create_job(
            kind="Surface texture",
            requested_name="Late cancellation",
            default_name="Wall texture",
            stage="Applying generated texture (99%)",
            cancel_callback=lambda: True,
        )
        self.assertTrue(self.manager.cancel_job(managed_job.job_id))
        self.workspace._cancelled_generation_job_ids.add(managed_job.job_id)

        self.workspace._handle_generation_worker_finished(managed_job.job_id)

        finished_job = self.manager.get_job(managed_job.job_id)
        self.assertIsNotNone(finished_job)
        assert finished_job is not None
        self.assertEqual(finished_job.status, JOB_STATUS_CANCELLED)
        self.assertEqual(finished_job.stage, "Cancelled")

    def test_finished_worker_cannot_leave_a_job_running_forever(self) -> None:
        managed_job = self.manager.create_job(
            kind="Surface texture",
            requested_name="Malformed result",
            default_name="Wall texture",
            stage="Applying generated texture (99%)",
        )

        self.workspace._handle_generation_worker_finished(managed_job.job_id)

        finished_job = self.manager.get_job(managed_job.job_id)
        self.assertIsNotNone(finished_job)
        assert finished_job is not None
        self.assertEqual(finished_job.status, JOB_STATUS_FAILED)
        self.assertIn("before its result", finished_job.stage)

    def test_late_progress_cannot_overwrite_a_terminal_job_or_tab_status(
        self,
    ) -> None:
        managed_job = self.manager.create_job(
            kind="Surface texture",
            requested_name="Finished wall",
            default_name="Wall texture",
            stage="Generating (80%)",
        )
        self.workspace._generation_workers[managed_job.job_id] = (  # type: ignore[assignment]
            object()
        )
        self.manager.complete_job(managed_job.job_id)
        self.workspace.status_label.setText("Committed texture")

        self.workspace._handle_generation_job_progress(
            managed_job.job_id,
            "Late provider callback (25%)",
        )

        terminal_job = self.manager.get_job(managed_job.job_id)
        self.assertIsNotNone(terminal_job)
        assert terminal_job is not None
        self.assertEqual(terminal_job.status, JOB_STATUS_COMPLETED)
        self.assertEqual(terminal_job.stage, "Completed")
        self.assertEqual(terminal_job.progress, 100)
        self.assertEqual(self.workspace.status_label.text(), "Committed texture")

    def test_shutdown_discards_output_saved_while_waiting_for_worker(self) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        job_id = "shutdown-save-race"
        request = _request(_FIRST_WALL)
        worker = SurfaceTextureWorker(
            job_id,
            self.provider,
            request,
            asset_directory,
        )
        asset_name = "shutdown-save-race.texture-512.png"
        asset_path = asset_directory / asset_name

        def inject_output() -> None:
            asset_directory.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(b"late worker output")
            saved_output = _SavedSurfaceTextureOutput(
                surface_ids=(_FIRST_WALL,),
                assignment_id="shutdown-save-race",
                variants=(
                    SurfaceTextureVariant(
                        resolution=512,
                        asset_path=asset_name,
                    ),
                ),
                texture_png_by_resolution=((512, b"late worker output"),),
            )
            with worker._output_lock:
                worker._unclaimed_saved_outputs = (saved_output,)

        thread = _LateOutputThread(inject_output)
        worker.succeeded.connect(
            self.workspace._handle_generation_job_succeeded
        )
        worker.failed.connect(self.workspace._handle_generation_job_failed)
        worker.progress.connect(self.workspace._handle_generation_job_progress)
        worker.cancelled.connect(
            self.workspace._handle_generation_job_cancelled
        )
        self.workspace._generation_threads[job_id] = thread
        self.workspace._generation_workers[job_id] = worker
        self.workspace._generation_requests[job_id] = request

        self.workspace.shutdown()

        self.assertFalse(asset_path.exists())

    def test_shutdown_waits_for_a_slow_finite_local_worker_stage(self) -> None:
        entered_local_stage = threading.Event()
        release_local_stage = threading.Event()

        def prepare_slowly(request, result, cancel_event):
            entered_local_stage.set()
            if not release_local_stage.wait(timeout=5.0):
                raise RuntimeError("Slow local stage was not released")
            return _prepare_surface_texture_outputs(
                request,
                result,
                cancel_event,
            )

        self.provider.release[_FIRST_WALL].set()
        with patch(
            "housemaker.surface_texture_workspace."
            "_prepare_surface_texture_outputs",
            new=prepare_slowly,
        ):
            self.assertTrue(
                self.workspace._start_generation(_request(_FIRST_WALL))
            )
            self.assertTrue(entered_local_stage.wait(timeout=2.0))
            active_thread = self.workspace._generation_thread
            self.assertIsNotNone(active_thread)
            release_timer = threading.Timer(
                0.5,
                release_local_stage.set,
            )
            release_timer.daemon = True
            release_timer.start()

            self.workspace.shutdown()

            release_timer.join(timeout=1.0)

        self.assertTrue(release_local_stage.is_set())
        assert active_thread is not None
        self.assertFalse(active_thread.isRunning())
        self.assertEqual(self.workspace._generation_threads, {})
        self.assertEqual(self.workspace._generation_workers, {})

    def test_custom_and_blank_names_persist_and_keep_generic_fallbacks(self) -> None:
        self.assertTrue(
            self.workspace._start_generation(
                _request(_FIRST_WALL, "Kitchen stone")
            )
        )
        self.assertTrue(self.workspace._start_generation(_request(_SECOND_WALL)))
        jobs_by_target = {
            request.surface_ids[0]: self.manager.get_job(job_id)
            for job_id, request in self.workspace._generation_requests.items()
        }
        self.assertEqual(jobs_by_target[_FIRST_WALL].name, "Kitchen stone")
        self.assertEqual(jobs_by_target[_SECOND_WALL].name, "Wall texture")
        self.provider.release_all()
        _wait_until(lambda: not self.workspace.is_generating)

        assignments_by_target = {
            assignment.surface_ids[0]: assignment
            for assignment in self.workspace.get_data().assignments
        }
        self.assertEqual(
            assignments_by_target[_FIRST_WALL].display_name,
            "Kitchen stone",
        )
        self.assertEqual(assignments_by_target[_SECOND_WALL].display_name, "")
        restored = SurfaceTextureData.from_dict(
            self.workspace.get_data().to_dict()
        )
        self.assertEqual(
            {assignment.display_name for assignment in restored.assignments},
            {"", "Kitchen stone"},
        )

        self.workspace.surface_view.set_selected_surface_ids((_FIRST_WALL,))
        self.workspace._refresh_texture_atlases()
        self.assertTrue(
            all(
                entry.display_name.startswith("Kitchen stone - ")
                for entry in self.workspace.texture_view.entries
            )
        )
        blank_item = next(
            self.workspace.other_texture_list.item(index)
            for index in range(self.workspace.other_texture_list.count())
            if "Wall texture - "
            in self.workspace.other_texture_list.item(index).text()
        )
        self.assertIn("Wall texture - ", blank_item.text())


if __name__ == "__main__":
    unittest.main()
