"""Local reference phases retain per-blob ownership and cancellation safety."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from typing import ClassVar
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import test_all_reference_generation as batch_fixtures
from PySide6.QtCore import QObject, Signal

from housemaker.plan_correction_models import (
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)
from housemaker.reference_generation_models import (
    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
)
from housemaker.reference_symmetry_inference import (
    SYMMETRY_MODEL_QWEN,
    SymmetryInferenceResult,
)


# ### Controlled local phase worker ###
class _ControlledLocalReferenceBatchThread(QObject):
    """Expose phase signals without loading models or making API requests."""

    item_progress = Signal(str, str, int)
    reference_ready = Signal(str, object)
    item_finished = Signal(str, object, str, bool)
    finished = Signal()
    created: ClassVar[list[_ControlledLocalReferenceBatchThread]] = []

    def __init__(
        self, items, model, symmetry_model, *, api_key, prompt, strength,
        negative_prompt, parent=None,
    ) -> None:
        super().__init__(parent)
        self.items = tuple(items)
        self.model = model
        self.symmetry_model = symmetry_model
        self.api_key = api_key
        self.prompt = prompt
        self.strength = strength
        self.negative_prompt = negative_prompt
        self.started = False
        self.running = False
        self.interrupted = False
        self.cancelled_items: list[str] = []
        self.created.append(self)

    def start(self) -> None:
        self.started = True
        self.running = True

    def requestInterruption(self) -> None:
        self.interrupted = True

    def cancel_item(self, job_id: str) -> None:
        self.cancelled_items.append(job_id)

    def isRunning(self) -> bool:
        return self.running

    def wait(self, *_args) -> bool:
        self.running = False
        return True

    def render(self, index: int, value: int) -> None:
        self.reference_ready.emit(self.items[index].job_id, batch_fixtures._reference(value))

    def decide(self, index: int, symmetric: bool, *, error: str = "") -> None:
        decision = None if error else SymmetryInferenceResult(
            symmetric=symmetric, model=self.symmetry_model,
            confidence=0.9, reason="Approximate bilateral geometry.",
        )
        self.item_finished.emit(self.items[index].job_id, decision, error, False)

    def complete(self) -> None:
        self.running = False
        self.finished.emit()


# ### Local phased batch integration ###
class LocalReferenceBatchWorkspaceTests(unittest.TestCase):
    _paint = batch_fixtures.AllReferenceGenerationTests._paint
    _seed_sources = batch_fixtures.AllReferenceGenerationTests._seed_sources

    def setUp(self) -> None:
        batch_fixtures.AllReferenceGenerationTests.setUp(self)
        _ControlledLocalReferenceBatchThread.created.clear()
        self.local_thread_patch = patch(
            "housemaker.merged_generation_workspace._LocalReferenceBatchThread",
            _ControlledLocalReferenceBatchThread, create=True,
        )
        self.local_thread_patch.start()
        self.workspace.set_reference_generation_model(PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1)
        self.workspace.set_object_symmetry_inference(True, SYMMETRY_MODEL_QWEN)

    def tearDown(self) -> None:
        try:
            batch_fixtures.AllReferenceGenerationTests.tearDown(self)
        finally:
            self.local_thread_patch.stop()
            _ControlledLocalReferenceBatchThread.created.clear()

    def _start_local(self) -> _ControlledLocalReferenceBatchThread:
        self.assertTrue(self.workspace.generate_all_references_button.isEnabled())
        self.workspace.generate_all_references_button.click()
        self.assertEqual(len(_ControlledLocalReferenceBatchThread.created), 1)
        worker = _ControlledLocalReferenceBatchThread.created[-1]
        self.assertTrue(worker.started)
        self.assertEqual(len(worker.items), 3)
        return worker

    def _finish_local(self, worker) -> None:
        for index, value in enumerate((101, 102, 103)):
            worker.render(index, value)
        for index, decision in enumerate((True, False, True)):
            worker.decide(index, decision)
        worker.complete()

    def test_local_queue_uses_one_worker_and_snapshots_every_input(self) -> None:
        self._seed_sources()
        selected_key = self.objects.current_inpaint_reference_key()
        expected_prompt = self.workspace._selected_reference_generation_prompt()
        worker = self._start_local()
        self.assertFalse(batch_fixtures._ControlledReferenceThread.created)
        self.assertEqual(worker.model, PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1)
        self.assertEqual(worker.symmetry_model, SYMMETRY_MODEL_QWEN)
        self.assertEqual(worker.prompt, expected_prompt)
        self.assertEqual([item.job_id for item in worker.items], [job.job_id for job in self.manager.jobs()])
        for item, value in zip(worker.items, (11, 22, 33)):
            np.testing.assert_array_equal(item.source_bgra, batch_fixtures._reference(value))
        self.assertEqual(self.objects.current_inpaint_reference_key(), selected_key)
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        self._finish_local(worker)
        self.assertTrue(all(job.status == "completed" for job in self.manager.jobs()))

    def test_rendered_images_appear_before_symmetry_and_decisions_stay_with_each_blob(self) -> None:
        keys = tuple(key for key, _source in self.objects.snapshot_inpaint_reference_inputs())
        worker = self._start_local()
        self.workspace.next_inpaint_button.click()
        self.workspace.next_inpaint_button.click()
        for index, value in enumerate((101, 102, 103)):
            worker.render(index, value)
        for key, value in zip(keys, (101, 102, 103)):
            np.testing.assert_array_equal(self.objects._inpaint_references[key], batch_fixtures._reference(value))
        self.assertTrue(all(job.status == "running" for job in self.manager.jobs()))
        self.assertEqual([self.objects.get_inpaint_symmetric_division(key) for key in keys], [None, None, None])
        for index, decision in enumerate((True, False, True)):
            worker.decide(index, decision)
        worker.complete()
        self.assertEqual([self.objects.get_inpaint_symmetric_division(key) for key in keys], [True, False, True])
        np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), batch_fixtures._reference(103))
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())

    def test_cancel_one_item_does_not_interrupt_shared_worker_or_other_items(self) -> None:
        worker = self._start_local()
        jobs = self.manager.jobs()
        self.assertTrue(self.manager.cancel_job(jobs[1].job_id))
        self.assertIn(jobs[1].job_id, worker.cancelled_items)
        self.assertFalse(worker.interrupted)
        self._finish_local(worker)
        self.assertEqual([job.status for job in self.manager.jobs()], ["completed", "cancelled", "completed"])
        keys = self.objects._current_inpaint_keys()
        self.assertIsNone(self.objects.get_inpaint_symmetric_division(keys[1]))

    def test_newly_pasted_source_is_not_overwritten_by_older_render_or_decision(self) -> None:
        worker = self._start_local()
        replacement = batch_fixtures._reference(77)
        self.workspace._activate_temporary_object_reference(replacement, source_label="New pasted reference")
        self._finish_local(worker)
        np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), replacement)
        self.assertEqual(self.manager.jobs()[0].status, "cancelled")
        self.assertFalse(self.objects.symmetric_division_checkbox.isChecked())

    def test_new_source_after_render_discards_late_symmetry_without_erasing_new_image(self) -> None:
        worker = self._start_local()
        worker.render(0, 101)
        replacement = batch_fixtures._reference(77)
        self.workspace._activate_temporary_object_reference(replacement, source_label="Replacement after render")
        worker.decide(0, True)
        for index in (1, 2):
            worker.render(index, 101 + index)
            worker.decide(index, False)
        worker.complete()
        np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), replacement)
        self.assertFalse(self.objects.symmetric_division_checkbox.isChecked())
        self.assertEqual(self.manager.jobs()[0].status, "cancelled")

    def test_deleting_blob_while_rendering_cannot_restore_it(self) -> None:
        worker = self._start_local()
        deleted_key = self.objects.current_inpaint_reference_key()
        self.workspace.clear_mask_button.click()
        self._finish_local(worker)
        self.assertNotIn(deleted_key, self.objects._inpaint_references)
        self.assertEqual(len(self.objects.inpaint_queue()), 2)
        self.assertEqual(self.manager.jobs()[0].status, "cancelled")

    def test_symmetry_failure_keeps_generated_reference_and_reports_only_that_item(self) -> None:
        self.objects.symmetric_division_checkbox.setChecked(True)
        worker = self._start_local()
        for index in range(3):
            worker.render(index, 101 + index)
        worker.decide(0, False, error="The selected symmetry model is unavailable")
        worker.decide(1, False)
        worker.decide(2, True)
        worker.complete()
        np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), batch_fixtures._reference(101))
        self.assertTrue(self.objects.symmetric_division_checkbox.isChecked())
        self.assertEqual([job.status for job in self.manager.jobs()], ["failed", "completed", "completed"])
        self.assertIn("unavailable", self.manager.jobs()[0].stage)

    def test_render_failure_does_not_prevent_remaining_references_or_symmetry(self) -> None:
        worker = self._start_local()
        worker.item_finished.emit(worker.items[0].job_id, None, "The image model could not render this object", False)
        for index in (1, 2):
            worker.render(index, 101 + index)
            worker.decide(index, True)
        worker.complete()
        self.assertEqual([job.status for job in self.manager.jobs()], ["failed", "completed", "completed"])
        self.assertIsNone(self.objects.get_temporary_object_reference())
        self.assertIn("could not render", self.manager.jobs()[0].stage)

    def test_controls_stay_disabled_until_shared_worker_finishes_model_cleanup(self) -> None:
        worker = self._start_local()
        for index in range(3):
            worker.render(index, 101 + index)
            worker.decide(index, True)
        self.assertTrue(all(job.status == "completed" for job in self.manager.jobs()))
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        self.assertFalse(self.workspace.reference_generation_view_combo.isEnabled())
        worker.complete()
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        self.assertTrue(self.workspace.reference_generation_view_combo.isEnabled())

    def test_disabled_inference_keeps_manual_choices_and_accepts_all_local_references(self) -> None:
        keys = self.objects._current_inpaint_keys()
        for key, enabled in zip(keys, (True, False, True)):
            self.assertTrue(self.objects.set_inpaint_symmetric_division(key, enabled))
        self.workspace.set_object_symmetry_inference(False, SYMMETRY_MODEL_QWEN)
        worker = self._start_local()
        self.assertIsNone(worker.symmetry_model)
        self.workspace.set_object_symmetry_inference(True, SYMMETRY_MODEL_QWEN)
        for index, value in enumerate((101, 102, 103)):
            worker.render(index, value)
            worker.item_finished.emit(worker.items[index].job_id, None, "", False)
        worker.complete()
        self.assertIsNone(worker.symmetry_model)
        self.assertEqual([self.objects.get_inpaint_symmetric_division(key) for key in keys], [True, False, True])
        self.assertEqual([job.status for job in self.manager.jobs()], ["completed"] * 3)
        for key, value in zip(keys, (101, 102, 103)):
            np.testing.assert_array_equal(self.objects._inpaint_references[key], batch_fixtures._reference(value))

    def test_active_local_batch_keeps_its_enabled_symmetry_snapshot(self) -> None:
        self.workspace.set_object_symmetry_inference(True, SYMMETRY_MODEL_QWEN)
        worker = self._start_local()
        self.workspace.set_object_symmetry_inference(False, SYMMETRY_MODEL_QWEN)
        self.assertEqual(worker.symmetry_model, SYMMETRY_MODEL_QWEN)
        self._finish_local(worker)
        keys = self.objects._current_inpaint_keys()
        self.assertEqual([self.objects.get_inpaint_symmetric_division(key) for key in keys], [True, False, True])
        self.assertEqual([job.status for job in self.manager.jobs()], ["completed"] * 3)

    def test_cancel_all_interrupts_worker_and_discards_late_signals(self) -> None:
        worker = self._start_local()
        self.workspace.generate_all_references_button.click()
        self.assertTrue(worker.interrupted)
        self._finish_local(worker)
        self.assertTrue(all(job.status == "cancelled" for job in self.manager.jobs()))
        self.assertFalse(self.objects._inpaint_references)

    def test_switching_projects_cancels_shared_worker_and_ignores_late_results(self) -> None:
        worker = self._start_local()
        self.objects.set_data(None)
        self.workspace.sync_shared_controls()
        self.assertTrue(worker.interrupted)
        self._finish_local(worker)
        self.assertTrue(all(job.status == "cancelled" for job in self.manager.jobs()))
        self.assertFalse(self.objects._inpaint_references)

    def test_shutdown_waits_for_shared_worker_and_cancels_all_rows(self) -> None:
        worker = self._start_local()
        self.workspace.shutdown()
        self.assertTrue(worker.interrupted)
        self.assertFalse(worker.running)
        self.assertTrue(all(job.status == "cancelled" for job in self.manager.jobs()))
        self._finish_local(worker)
        self.assertFalse(self.objects._inpaint_references)

    def test_remote_provider_keeps_existing_per_request_scheduler(self) -> None:
        self.workspace.set_reference_generation_model(PLAN_CORRECTION_MODEL_GPT_IMAGE_2)
        workers = batch_fixtures.AllReferenceGenerationTests._start_all(self)
        self.assertFalse(_ControlledLocalReferenceBatchThread.created)
        self.assertEqual([worker.started for worker in workers], [True, False, False])
        batch_fixtures.AllReferenceGenerationTests._finish_all(self, workers)
        self.assertTrue(all(job.status == "completed" for job in self.manager.jobs()))

    def test_sdxl_uses_the_same_grouped_local_route(self) -> None:
        self.workspace.set_reference_generation_model(REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION)
        worker = self._start_local()
        self.assertEqual(worker.model, REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION)
        self.assertFalse(batch_fixtures._ControlledReferenceThread.created)
        self._finish_local(worker)
        self.assertTrue(all(job.status == "completed" for job in self.manager.jobs()))


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
