"""Cross-frame reference batches without paid requests or model inference."""

# ### Environment and imports ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
from PySide6.QtWidgets import QApplication

from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.merged_generation_workspace import MergedGenerationWorkspace
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace

# ### Controlled provider fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _reference(value: int) -> np.ndarray:
    image = np.full((20, 24, 4), value, dtype=np.uint8)
    image[:, :, 3] = 255
    return image


class _ControlledReferenceThread(QObject):
    """Expose each queued request while completing workers only on demand."""

    progress_changed = Signal(str, int)
    finished = Signal()
    created: ClassVar[list[_ControlledReferenceThread]] = []

    def __init__(
        self, source_bgra, prompt, model, *, api_key, strength,
        negative_prompt, parent=None,
    ) -> None:
        super().__init__(parent)
        self.source_bgra = source_bgra.copy()
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.strength = strength
        self.negative_prompt = negative_prompt
        self.result = None
        self.error_message = None
        self.was_cancelled = False
        self.interrupted = False
        self.started = False
        self.running = False
        self.created.append(self)

    def start(self) -> None:
        self.started = True
        self.running = True

    def requestInterruption(self) -> None:
        self.interrupted = True

    def isRunning(self) -> bool:
        return self.running

    def wait(self, *_args) -> bool:
        self.running = False
        return True

    def complete(self, image=None, *, error=None) -> None:
        self.result = image
        self.error_message = error
        self.running = False
        self.finished.emit()


# ### Cross-frame batch integration tests ###
class AllReferenceGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manager = GenerationJobManager()
        self.objects = GenerationWorkspace(
            asset_directory=self.root / "objects", job_manager=self.manager,
        )
        self.surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=self.root / "surfaces", job_manager=self.manager,
            shared_controls=self.objects.get_shared_controls(),
        )
        self.workspace = MergedGenerationWorkspace(
            self.surfaces, self.objects, self.manager,
        )
        _ControlledReferenceThread.created.clear()
        self.thread_patch = patch(
            "housemaker.merged_generation_workspace._ReferenceGenerationThread",
            _ControlledReferenceThread,
        )
        self.thread_patch.start()
        self.video = self.root / "references.avi"
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48),
        )
        self.assertTrue(writer.isOpened())
        try:
            for value in (40, 80, 120):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        self.objects.load_video(str(self.video))
        self.surfaces.load_video(str(self.video))
        self._paint(0, 0.25, 0.75)
        self._paint(2, 0.5)
        self.workspace.seekbar.setValue(0)
        self.workspace.video_view.select_blob(0)
        self.workspace.sync_shared_controls()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.objects.shutdown()
        self.surfaces.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()
        self.thread_patch.stop()
        _ControlledReferenceThread.created.clear()
        self.temporary.cleanup()

    def _paint(self, frame: int, *positions: float) -> None:
        self.workspace.seekbar.setValue(frame)
        self.workspace.video_view.set_strokes([
            MaskStroke(MASK_MODE_PAINT, 0.07, (MaskPoint(x, 0.5),))
            for x in positions
        ])
        self.workspace.video_view.strokes_changed.emit(
            self.workspace.video_view.get_strokes(),
        )

    def _seed_sources(self) -> None:
        for value in (11, 22, 33):
            self.workspace._activate_temporary_object_reference(
                _reference(value), source_label="Pasted test reference",
            )
            self.workspace.next_inpaint_button.click()

    def _start_all(self) -> list[_ControlledReferenceThread]:
        before = len(_ControlledReferenceThread.created)
        self.assertTrue(self.workspace.generate_all_references_button.isEnabled())
        self.workspace.generate_all_references_button.click()
        workers = _ControlledReferenceThread.created[before:]
        self.assertEqual(len(workers), 3)
        return workers

    def _finish_all(self, workers, values=(101, 102, 103)) -> None:
        for worker, value in zip(workers, values):
            self.assertTrue(worker.started)
            worker.complete(_reference(value))

    def test_button_is_below_single_reference_and_requires_queued_blobs(self) -> None:
        layout = self.workspace.reference_generation_section.layout()
        self.assertEqual(
            layout.indexOf(self.workspace.generate_all_references_button),
            layout.indexOf(self.workspace.generate_reference_button) + 1,
        )
        self.assertEqual(
            self.workspace.generate_all_references_button.text(),
            "Generate reference for all",
        )
        self.objects.set_data(None)
        self.workspace.sync_shared_controls()
        self.assertFalse(self.workspace.generate_all_references_button.isEnabled())

    def test_creates_every_job_but_runs_serially_without_changing_selection(self) -> None:
        self._seed_sources()
        original_key = self.objects.current_inpaint_reference_key()
        expected_prompt = self.workspace._selected_reference_generation_prompt()
        expected_model = self.workspace._reference_generation_model
        workers = self._start_all()
        self.assertEqual(len(self.manager.jobs()), 3)
        self.assertEqual([worker.started for worker in workers], [True, False, False])
        self.assertEqual(self.objects.current_inpaint_reference_key(), original_key)
        self.assertEqual(self.objects._displayed_frame_index, 0)
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        for worker, value in zip(workers, (11, 22, 33)):
            np.testing.assert_array_equal(worker.source_bgra, _reference(value))
            self.assertEqual(worker.prompt, expected_prompt)
            self.assertEqual(worker.model, expected_model)
        self._finish_all(workers)
        self.assertTrue(all(job.status == "completed" for job in self.manager.jobs()))

    def test_navigation_does_not_cancel_and_results_stay_with_each_blob(self) -> None:
        self._seed_sources()
        workers = self._start_all()
        self.workspace.next_inpaint_button.click()
        self.workspace.next_inpaint_button.click()
        self.assertEqual(self.objects._displayed_frame_index, 2)
        self.assertFalse(any(worker.interrupted for worker in workers))
        self._finish_all(workers)
        np.testing.assert_array_equal(
            self.objects.get_temporary_object_reference(), _reference(103),
        )
        self.workspace.next_inpaint_button.click()
        for value in (101, 102, 103):
            np.testing.assert_array_equal(
                self.objects.get_temporary_object_reference(), _reference(value),
            )
            np.testing.assert_array_equal(
                self.surfaces.get_temporary_reference(), _reference(value),
            )
            self.workspace.next_inpaint_button.click()

    def test_model_failure_does_not_stop_other_references(self) -> None:
        self._seed_sources()
        workers = self._start_all()
        workers[0].complete(error="Provider could not generate this object")
        self.assertTrue(workers[1].started)
        workers[1].complete(_reference(102))
        workers[2].complete(_reference(103))
        self.assertEqual(
            [job.status for job in self.manager.jobs()],
            ["failed", "completed", "completed"],
        )
        np.testing.assert_array_equal(
            self.objects.get_temporary_object_reference(), _reference(11),
        )

    def test_unavailable_source_fails_only_affected_job_and_continues_queue(self) -> None:
        inputs = self.objects.snapshot_inpaint_reference_inputs()
        affected_key = inputs[0][0]
        read_input = self.objects.get_inpaint_reference_input

        def checked_input(key):
            if key == affected_key:
                raise ValueError("The source video frame could not be decoded")
            return read_input(key)

        with (
            patch.object(self.objects, "snapshot_inpaint_reference_inputs", return_value=inputs),
            patch.object(self.objects, "get_inpaint_reference_input", side_effect=checked_input),
        ):
            workers = self._start_all()
            self.assertFalse(workers[0].started)
            self.assertTrue(workers[1].started)
            workers[1].complete(_reference(102))
            workers[2].complete(_reference(103))
        self.assertEqual(
            [job.status for job in self.manager.jobs()],
            ["failed", "completed", "completed"],
        )
        self.assertIn("could not be decoded", self.manager.jobs()[0].stage)
        self.assertNotIn(affected_key, self.objects._inpaint_references)

    def test_worker_start_failure_does_not_block_other_queued_jobs(self) -> None:
        start = _ControlledReferenceThread.start

        def checked_start(worker):
            if worker is _ControlledReferenceThread.created[0]:
                raise RuntimeError("The reference worker could not start")
            start(worker)

        with patch.object(_ControlledReferenceThread, "start", checked_start):
            workers = self._start_all()
            self.assertFalse(workers[0].started)
            self.assertTrue(workers[1].started)
            workers[1].complete(_reference(102))
            workers[2].complete(_reference(103))
        self.assertEqual(
            [job.status for job in self.manager.jobs()],
            ["failed", "completed", "completed"],
        )
        self.assertIn("could not start", self.manager.jobs()[0].stage)

    def test_opening_component_mode_preserves_component_reference_and_blob_results(self) -> None:
        inputs = self.objects.snapshot_inpaint_reference_inputs()
        workers = self._start_all()
        self.assertTrue(self.objects.set_door_slot_editing_target(
            door_id="door-one", slot_id="body", object_id="door-one-body",
            object_name="Door one", slot_display_name="Body",
        ))
        component_reference = _reference(77)
        self.workspace._activate_temporary_object_reference(
            component_reference, source_label="Door body reference",
        )
        self._finish_all(workers)
        np.testing.assert_array_equal(
            self.objects.get_temporary_object_reference(), component_reference,
        )
        np.testing.assert_array_equal(
            self.surfaces.get_temporary_reference(), component_reference,
        )
        for (key, _source), value in zip(inputs, (101, 102, 103)):
            np.testing.assert_array_equal(
                self.objects._inpaint_references[key], _reference(value),
            )
        self.assertTrue(all(job.status == "completed" for job in self.manager.jobs()))
        self.assertEqual(self.objects._door_slot_editing_target.door_id, "door-one")

    def test_cancelling_one_queued_job_does_not_cancel_the_batch(self) -> None:
        workers = self._start_all()
        jobs = self.manager.jobs()
        self.assertTrue(self.manager.cancel_job(jobs[1].job_id))
        workers[0].complete(_reference(101))
        self.assertFalse(workers[1].started)
        self.assertTrue(workers[2].started)
        workers[2].complete(_reference(103))
        self.assertEqual(
            [job.status for job in self.manager.jobs()],
            ["completed", "cancelled", "completed"],
        )

    def test_cancelling_active_job_waits_for_worker_before_starting_next(self) -> None:
        workers = self._start_all()
        self.assertTrue(self.manager.cancel_job(self.manager.jobs()[0].job_id))
        self.assertTrue(workers[0].interrupted)
        self.assertFalse(workers[1].started)
        workers[0].complete(_reference(101))
        self.assertTrue(workers[1].started)
        self.assertIsNone(self.objects.get_temporary_object_reference())
        workers[1].complete(_reference(102))
        workers[2].complete(_reference(103))
        self.assertEqual(
            [job.status for job in self.manager.jobs()],
            ["cancelled", "completed", "completed"],
        )

    def test_single_reference_and_batch_requests_are_mutually_exclusive(self) -> None:
        self.workspace.generate_reference_button.click()
        single = _ControlledReferenceThread.created[-1]
        self.assertFalse(self.workspace.generate_all_references_button.isEnabled())
        single.complete(_reference(80))
        workers = self._start_all()
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        self._finish_all(workers)
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())

    def test_queued_requests_keep_their_original_prompt_and_options(self) -> None:
        workers = self._start_all()
        settings = [
            (worker.prompt, worker.model, worker.strength, worker.negative_prompt)
            for worker in workers
        ]
        self.workspace.reference_generation_prompt_edit.setText("New prompt")
        self.workspace.reference_generation_strength_spinbox.setValue(0.8)
        self.workspace.reference_generation_negative_prompt_edit.setText("New negative")
        self._finish_all(workers)
        self.assertEqual(
            [
                (worker.prompt, worker.model, worker.strength, worker.negative_prompt)
                for worker in workers
            ],
            settings,
        )

    def test_cancelling_all_discards_late_results(self) -> None:
        workers = self._start_all()
        self.workspace.generate_all_references_button.click()
        workers[0].complete(_reference(101))
        self.assertTrue(workers[0].interrupted)
        self.assertFalse(any(worker.started for worker in workers[1:]))
        self.assertTrue(all(job.status == "cancelled" for job in self.manager.jobs()))
        self.assertFalse(self.objects._inpaint_references)
        self.assertIsNone(self.objects.get_temporary_object_reference())

    def test_deleting_blob_while_running_does_not_restore_its_reference(self) -> None:
        workers = self._start_all()
        deleted_key = self.objects.current_inpaint_reference_key()
        self.workspace.clear_mask_button.click()
        workers[0].complete(_reference(101))
        self.assertNotIn(deleted_key, self.objects._inpaint_references)
        self.assertEqual(len(self.objects.inpaint_queue()), 2)
        self.assertEqual(self.manager.jobs()[0].status, "cancelled")
        for worker in workers[1:]:
            if worker.started:
                worker.complete(_reference(102))

    def test_pasting_new_source_does_not_get_overwritten_by_old_job(self) -> None:
        workers = self._start_all()
        replacement = _reference(77)
        self.workspace._activate_temporary_object_reference(
            replacement, source_label="Replacement while generating",
        )
        workers[0].complete(_reference(101))
        np.testing.assert_array_equal(
            self.objects.get_temporary_object_reference(), replacement,
        )
        self.assertEqual(self.manager.jobs()[0].status, "cancelled")
        self.assertTrue(workers[1].started)
        workers[1].complete(_reference(102))
        workers[2].complete(_reference(103))

    def test_repeated_batch_retries_original_sources_not_generated_results(self) -> None:
        self._seed_sources()
        workers = self._start_all()
        self._finish_all(workers)
        retry_workers = self._start_all()
        for worker, value in zip(retry_workers, (11, 22, 33)):
            np.testing.assert_array_equal(worker.source_bgra, _reference(value))
        self._finish_all(retry_workers, values=(201, 202, 203))

    def test_switching_projects_cancels_jobs_and_cannot_leak_references(self) -> None:
        workers = self._start_all()
        self.objects.set_data(None)
        self.workspace.sync_shared_controls()
        workers[0].complete(_reference(101))
        self.assertTrue(all(job.status == "cancelled" for job in self.manager.jobs()))
        self.assertFalse(self.objects._inpaint_references)
        self.assertIsNone(self.objects.get_temporary_object_reference())
        self.assertFalse(any(worker.started for worker in workers[1:]))

    def test_shutdown_cancels_running_and_queued_reference_jobs(self) -> None:
        workers = self._start_all()
        self.workspace.shutdown()
        self.assertTrue(workers[0].interrupted)
        self.assertFalse(workers[0].running)
        self.assertFalse(any(worker.started for worker in workers[1:]))
        self.assertTrue(all(job.status == "cancelled" for job in self.manager.jobs()))
        self.assertFalse(self.workspace.generate_all_references_button.isEnabled())
        workers[0].complete(_reference(101))
        self.assertFalse(self.objects._inpaint_references)


# ### Standalone test runner ###
if __name__ == "__main__":
    unittest.main()
