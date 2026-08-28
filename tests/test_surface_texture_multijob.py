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
    _decode_png_rgba,
    _encode_png,
    _encode_rgba_png,
    _prepare_localized_surface_texture_request,
    _prepare_surface_texture_outputs,
    rasterize_texture_mask_strokes,
)
from tests.test_surface_texture_workspace import (
    _colored_texture_png,
    _set_current_mask,
    _surface_assignment_with_variants,
    _test_level,
    _test_stroke,
    _texture_png,
    _write_test_video,
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

    def test_localized_preflight_is_nonblocking_and_rejects_changed_base(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "localized-base",
            (_FIRST_WALL,),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(_FIRST_WALL,),
                texture_mask_strokes={_FIRST_WALL: [_test_stroke(0.4)]},
                assignments=[assignment],
            )
        )
        video_path = self._temporary_path / "localized-preflight.avi"
        _write_test_video(video_path, frame_count=1)
        self.workspace.load_video(str(video_path))
        _set_current_mask(self.workspace, _test_stroke(0.2))
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="test-key")
        )
        entered_preflight = threading.Event()
        release_preflight = threading.Event()

        def prepare_slowly(request, sources, directory, cancel_event):
            entered_preflight.set()
            if not release_preflight.wait(timeout=1.0):
                raise RuntimeError("Localized preflight ran on the GUI thread")
            return _prepare_localized_surface_texture_request(
                request,
                sources,
                directory,
                cancel_event,
            )

        with patch(
            "housemaker.surface_texture_workspace."
            "_prepare_localized_surface_texture_request",
            new=prepare_slowly,
        ):
            started_at = time.monotonic()
            self.workspace.generate()
            elapsed = time.monotonic() - started_at
            try:
                self.assertLess(elapsed, 0.5)
                self.assertTrue(entered_preflight.wait(timeout=2.0))
                active_request = next(
                    iter(self.workspace._generation_requests.values())
                )
                self.assertIsNone(active_request.existing_texture_png)
                canonical_variant = assignment.texture_variant_for_resolution(
                    2048
                )
                self.assertIsNotNone(canonical_variant)
                assert canonical_variant is not None
                with (asset_directory / canonical_variant.asset_path).open(
                    "ab"
                ) as handle:
                    handle.write(b"changed-after-submit")
            finally:
                release_preflight.set()

            _wait_until(lambda: not self.workspace.is_generating)

        self.assertFalse(self.provider.started[_FIRST_WALL].is_set())
        managed_job = self.manager.jobs()[0]
        self.assertEqual(managed_job.status, JOB_STATUS_FAILED)
        self.assertIn("changed after this job", managed_job.stage)

    def test_localized_file_preflight_operations_run_off_the_gui_thread(
        self,
    ) -> None:
        asset_directory = self._temporary_path / "surface_assets"
        assignment = _surface_assignment_with_variants(
            asset_directory,
            "threaded-localized-base",
            (_FIRST_WALL,),
        )
        self.workspace.set_data(
            SurfaceTextureData(
                selected_surface_type="wall",
                selected_surface_ids=(_FIRST_WALL,),
                texture_mask_strokes={_FIRST_WALL: [_test_stroke(0.4)]},
                assignments=[assignment],
            )
        )
        video_path = self._temporary_path / "threaded-localized.avi"
        _write_test_video(video_path, frame_count=1)
        self.workspace.load_video(str(video_path))
        _set_current_mask(self.workspace, _test_stroke(0.2))
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="test-key")
        )
        canonical_variant = assignment.texture_variant_for_resolution(2048)
        self.assertIsNotNone(canonical_variant)
        assert canonical_variant is not None
        canonical_path = (
            asset_directory / canonical_variant.asset_path
        ).resolve()
        gui_thread_id = threading.get_ident()
        observed_threads: dict[str, list[int]] = {
            "read": [],
            "decode": [],
            "rasterize": [],
            "encode_rgba": [],
            "encode": [],
        }
        original_read_bytes = Path.read_bytes

        def tracked_read_bytes(path: Path) -> bytes:
            if path.resolve() == canonical_path:
                observed_threads["read"].append(threading.get_ident())
            return original_read_bytes(path)

        def tracked_decode(*args, **kwargs):
            observed_threads["decode"].append(threading.get_ident())
            return _decode_png_rgba(*args, **kwargs)

        def tracked_rasterize(*args, **kwargs):
            observed_threads["rasterize"].append(threading.get_ident())
            return rasterize_texture_mask_strokes(*args, **kwargs)

        def tracked_encode_rgba(*args, **kwargs):
            observed_threads["encode_rgba"].append(threading.get_ident())
            return _encode_rgba_png(*args, **kwargs)

        def tracked_encode(*args, **kwargs):
            observed_threads["encode"].append(threading.get_ident())
            return _encode_png(*args, **kwargs)

        with (
            patch.object(Path, "read_bytes", new=tracked_read_bytes),
            patch(
                "housemaker.surface_texture_workspace._decode_png_rgba",
                new=tracked_decode,
            ),
            patch(
                "housemaker.surface_texture_workspace."
                "rasterize_texture_mask_strokes",
                new=tracked_rasterize,
            ),
            patch(
                "housemaker.surface_texture_workspace._encode_rgba_png",
                new=tracked_encode_rgba,
            ),
            patch(
                "housemaker.surface_texture_workspace._encode_png",
                new=tracked_encode,
            ),
        ):
            self.workspace.generate()
            try:
                self.assertTrue(
                    self.provider.started[_FIRST_WALL].wait(timeout=5.0)
                )
                for operation, thread_ids in observed_threads.items():
                    with self.subTest(operation=operation):
                        self.assertTrue(thread_ids)
                        self.assertNotIn(gui_thread_id, thread_ids)
            finally:
                self.provider.release[_FIRST_WALL].set()
            _wait_until(lambda: not self.workspace.is_generating)

        self.assertEqual(self.manager.jobs()[0].status, JOB_STATUS_COMPLETED)

    def test_localized_worker_preserves_legacy_viewer_texture_fallback(
        self,
    ) -> None:
        video_path = self._temporary_path / "legacy-localized.avi"
        _write_test_video(video_path, frame_count=1)
        self.workspace.load_video(str(video_path))
        _set_current_mask(self.workspace, _test_stroke(0.2))
        self.workspace.surface_view.set_surface_texture(
            (_FIRST_WALL,),
            _texture_png(),
        )
        self.workspace.surface_view.set_selected_surface_ids((_FIRST_WALL,))
        self.workspace.surface_view.add_texture_mask_stroke(
            _FIRST_WALL,
            _test_stroke(0.4),
        )
        self.workspace.set_runtime_settings(
            GenerationServiceSettings(meshy_api_key="test-key")
        )
        self.provider.release[_FIRST_WALL].set()

        self.workspace.generate()
        _wait_until(lambda: not self.workspace.is_generating)

        self.assertEqual(len(self.provider.requests), 1)
        submitted_request = self.provider.requests[0]
        self.assertIsNotNone(submitted_request.existing_texture_png)
        self.assertIsNotNone(submitted_request.edit_mask_png)
        self.assertEqual(
            tuple(
                surface_id
                for surface_id, _mask_png
                in submitted_request.surface_edit_mask_pngs
            ),
            (_FIRST_WALL,),
        )
        self.assertEqual(self.manager.jobs()[0].status, JOB_STATUS_COMPLETED)

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
