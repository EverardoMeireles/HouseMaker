"""Symmetry inference uses generated images and stays bound to each blob."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import test_all_reference_generation as batch_fixtures
import test_reference_generation_workspace as settings_fixtures

from housemaker.generation_jobs import JOB_STATUS_CANCELLED, JOB_STATUS_FAILED
from housemaker.merged_generation_workspace import _ReferenceGenerationThread
from housemaker.reference_symmetry_inference import (
    DEFAULT_SYMMETRY_INFERENCE_MODEL,
    SYMMETRY_INFERENCE_MODEL_OPTIONS,
    SYMMETRY_MODEL_LUNA,
    SYMMETRY_MODEL_REFLECT3D,
    SymmetryInferenceResult,
)


# ### Controlled shared-workspace setup ###
class ReferenceSymmetryWorkspaceTests(unittest.TestCase):
    tearDown = batch_fixtures.AllReferenceGenerationTests.tearDown
    _paint = batch_fixtures.AllReferenceGenerationTests._paint
    _start_all = batch_fixtures.AllReferenceGenerationTests._start_all

    def setUp(self):
        batch_fixtures.AllReferenceGenerationTests.setUp(self)
        self.workspace.set_object_symmetry_inference(
            True, DEFAULT_SYMMETRY_INFERENCE_MODEL,
        )

    def _decision(self, symmetric, model=DEFAULT_SYMMETRY_INFERENCE_MODEL):
        return SymmetryInferenceResult(
            symmetric=symmetric, model=model, confidence=0.9,
            reason="Approximate bilateral geometry.", elapsed_seconds=0.25,
        )

    def test_model_selector_is_no_longer_in_generation_controls(self):
        self.assertFalse(hasattr(self.workspace, "symmetry_inference_model_combo"))
        self.assertFalse(hasattr(self.workspace, "symmetry_inference_model_control"))
        layout = self.workspace.division_options_section.layout()
        self.assertTrue(layout.itemAt(0).widget().isAncestorOf(self.objects.symmetric_division_checkbox))

    def test_single_job_checks_then_unchecks_after_inference(self):
        for decision in (True, False):
            self.workspace.generate_reference_button.click()
            worker = batch_fixtures._ControlledReferenceThread.created[-1]
            self.assertEqual(worker.symmetry_model, DEFAULT_SYMMETRY_INFERENCE_MODEL)
            worker.symmetry_result = self._decision(decision)
            worker.complete(batch_fixtures._reference(100 + int(decision)))
            self.assertEqual(self.objects.symmetric_division_checkbox.isChecked(), decision)
            self.assertTrue(self.objects.symmetric_division_checkbox.isEnabled())
            self.assertIn("Symmetric division", self.workspace.reference_generation_status_label.text())

    def test_batch_saves_independent_decisions_and_request_flags_across_frames(self):
        workers = self._start_all()
        keys = self.objects._current_inpaint_keys()
        for worker, decision, value in zip(workers, (True, False, True), (101, 102, 103)):
            worker.symmetry_result = self._decision(decision)
            worker.complete(batch_fixtures._reference(value))
        self.assertEqual([self.objects.get_inpaint_symmetric_division(key) for key in keys], [True, False, True])
        self.assertTrue(self.objects.symmetric_division_checkbox.isChecked())
        self.workspace.next_inpaint_button.click()
        self.assertFalse(self.objects.symmetric_division_checkbox.isChecked())
        self.workspace.next_inpaint_button.click()
        self.assertTrue(self.objects.symmetric_division_checkbox.isChecked())
        self.assertEqual([request.symmetric_division_enabled for request in self.objects._build_generation_requests()],
                         [True, False, True])

    def test_inference_failure_keeps_image_and_existing_flag_and_reports_error(self):
        self.objects.symmetric_division_checkbox.setChecked(True)
        self.workspace.generate_reference_button.click()
        worker = batch_fixtures._ControlledReferenceThread.created[-1]
        worker.symmetry_error = "The selected symmetry model is unavailable"
        worker.complete(batch_fixtures._reference(123))
        np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), batch_fixtures._reference(123))
        self.assertTrue(self.objects.symmetric_division_checkbox.isChecked())
        self.assertEqual(self.manager.jobs()[-1].status, JOB_STATUS_FAILED)
        self.assertIn("unavailable", self.workspace.reference_generation_status_label.text())

    def test_cancelled_late_symmetry_result_does_not_modify_flag(self):
        self.workspace.generate_reference_button.click()
        worker = batch_fixtures._ControlledReferenceThread.created[-1]
        self.workspace.generate_reference_button.click()
        worker.symmetry_result = self._decision(True)
        worker.complete(batch_fixtures._reference(123))
        self.assertFalse(self.objects.symmetric_division_checkbox.isChecked())
        self.assertEqual(self.manager.jobs()[-1].status, JOB_STATUS_CANCELLED)

    def test_batch_snapshots_settings_model_even_when_settings_change_during_generation(self):
        self.workspace.set_object_symmetry_inference(True, SYMMETRY_MODEL_LUNA)
        workers = self._start_all()
        self.assertTrue(all(worker.symmetry_model == SYMMETRY_MODEL_LUNA for worker in workers))
        self.workspace.set_object_symmetry_inference(False, DEFAULT_SYMMETRY_INFERENCE_MODEL)
        for worker in workers:
            worker.symmetry_result = self._decision(False, SYMMETRY_MODEL_LUNA)
            worker.complete(batch_fixtures._reference(123))
        self.assertTrue(all(worker.symmetry_model == SYMMETRY_MODEL_LUNA for worker in workers))
        self.assertEqual([job.status for job in self.manager.jobs()], ["completed"] * 3)

    def test_disabled_single_reference_does_not_change_manual_symmetric_division(self):
        self.workspace.set_object_symmetry_inference(False, SYMMETRY_MODEL_LUNA)
        self.objects.symmetric_division_checkbox.setChecked(True)
        self.workspace.generate_reference_button.click()
        worker = batch_fixtures._ControlledReferenceThread.created[-1]
        self.assertIsNone(worker.symmetry_model)
        worker.complete(batch_fixtures._reference(123))
        self.assertTrue(self.objects.symmetric_division_checkbox.isChecked())
        self.assertEqual(self.manager.jobs()[-1].status, "completed")
        np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), batch_fixtures._reference(123))

    def test_disabled_api_batch_keeps_individual_manual_division_choices(self):
        keys = self.objects._current_inpaint_keys()
        for key, enabled in zip(keys, (True, False, True)):
            self.assertTrue(self.objects.set_inpaint_symmetric_division(key, enabled))
        self.workspace.set_object_symmetry_inference(False, SYMMETRY_MODEL_LUNA)
        workers = self._start_all()
        self.assertTrue(all(worker.symmetry_model is None for worker in workers))
        self.workspace.set_object_symmetry_inference(True, DEFAULT_SYMMETRY_INFERENCE_MODEL)
        for worker, value in zip(workers, (101, 102, 103)):
            worker.complete(batch_fixtures._reference(value))
        self.assertEqual([self.objects.get_inpaint_symmetric_division(key) for key in keys], [True, False, True])
        self.assertTrue(all(worker.symmetry_model is None for worker in workers))
        self.assertEqual([job.status for job in self.manager.jobs()], ["completed"] * 3)

    def test_single_reference_freezes_settings_until_next_request(self):
        self.workspace.set_object_symmetry_inference(True, SYMMETRY_MODEL_LUNA)
        self.workspace.generate_reference_button.click()
        worker = batch_fixtures._ControlledReferenceThread.created[-1]
        self.workspace.set_object_symmetry_inference(False, DEFAULT_SYMMETRY_INFERENCE_MODEL)
        self.assertEqual(worker.symmetry_model, SYMMETRY_MODEL_LUNA)
        worker.symmetry_result = self._decision(True, SYMMETRY_MODEL_LUNA)
        worker.complete(batch_fixtures._reference(123))
        self.assertTrue(self.objects.symmetric_division_checkbox.isChecked())
        self.workspace.generate_reference_button.click()
        next_worker = batch_fixtures._ControlledReferenceThread.created[-1]
        self.assertIsNone(next_worker.symmetry_model)
        next_worker.complete(batch_fixtures._reference(124))
        self.assertTrue(self.objects.symmetric_division_checkbox.isChecked())


# ### Settings-to-reference integration ###
class ReferenceSymmetrySettingsIntegrationTests(unittest.TestCase):
    setUp = settings_fixtures.ReferenceGenerationWorkspaceTests.setUp
    tearDown = settings_fixtures.ReferenceGenerationWorkspaceTests.tearDown
    _load_video = settings_fixtures.ReferenceGenerationWorkspaceTests._load_video
    _start = settings_fixtures.ReferenceGenerationWorkspaceTests._start
    _reopened_workspace = settings_fixtures.ReferenceGenerationWorkspaceTests._reopened_workspace

    def _apply_reference_generation_model(self):
        settings = self.settings_widget.get_settings()
        self.workspace.set_reference_generation_model(settings.reference_generation_model)
        self.workspace.set_object_symmetry_inference(
            settings.enable_object_symmetry_inference, settings.symmetry_inference_model,
        )

    def test_fresh_settings_keep_inference_disabled_and_reflect3d_selected(self):
        self.assertFalse(self.workspace._enable_object_symmetry_inference)
        self.assertEqual(self.workspace._symmetry_inference_model, SYMMETRY_MODEL_REFLECT3D)
        self._load_video()
        self.objects.symmetric_division_checkbox.setChecked(True)
        worker = self._start()
        self.assertIsNone(worker.symmetry_model)
        worker.complete(settings_fixtures._opaque_reference(100))
        self.assertTrue(self.objects.symmetric_division_checkbox.isChecked())

    def test_settings_changes_apply_to_next_reference_without_changing_manual_division(self):
        self._load_video()
        combo = self.settings_widget.symmetry_inference_model_combo
        combo.setCurrentIndex(combo.findData(SYMMETRY_MODEL_LUNA))
        self.settings_widget.enable_object_symmetry_inference_checkbox.setChecked(False)
        self.objects.symmetric_division_checkbox.setChecked(True)
        first = self._start()
        self.assertIsNone(first.symmetry_model)
        first.complete(settings_fixtures._opaque_reference(101))
        self.assertTrue(self.objects.symmetric_division_checkbox.isChecked())
        self.settings_widget.enable_object_symmetry_inference_checkbox.setChecked(True)
        second = self._start()
        self.assertEqual(second.symmetry_model, SYMMETRY_MODEL_LUNA)
        second.symmetry_result = SymmetryInferenceResult(symmetric=False, model=SYMMETRY_MODEL_LUNA)
        second.complete(settings_fixtures._opaque_reference(102))
        self.assertFalse(self.objects.symmetric_division_checkbox.isChecked())

    def test_workspace_initialization_reads_persisted_settings_without_generation_selector(self):
        combo = self.settings_widget.symmetry_inference_model_combo
        combo.setCurrentIndex(combo.findData(SYMMETRY_MODEL_LUNA))
        self.settings_widget.enable_object_symmetry_inference_checkbox.setChecked(False)
        with self._reopened_workspace() as workspace:
            self.assertFalse(workspace._enable_object_symmetry_inference)
            self.assertEqual(workspace._symmetry_inference_model, SYMMETRY_MODEL_LUNA)
            self.assertFalse(hasattr(workspace, "symmetry_inference_model_combo"))


# ### The real worker's second inference stage (both providers mocked) ###
class ReferenceSymmetryThreadTests(unittest.TestCase):
    def test_disabled_inference_never_calls_model_and_keeps_generated_image(self):
        generated = batch_fixtures._reference(200)
        thread = _ReferenceGenerationThread(batch_fixtures._reference(10), "front", "image", api_key="secret", strength=0.5)
        self.assertIsNone(thread.symmetry_model)
        with patch("housemaker.reference_generation.generate_object_reference", return_value=generated), patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry",
        ) as infer:
            thread.run()
        infer.assert_not_called()
        self.assertIs(thread.result, generated)
        self.assertIsNone(thread.symmetry_result)
        self.assertIsNone(thread.symmetry_error)
        self.assertIsNone(thread.error_message)
        self.assertEqual(thread._api_key, "")
        thread.deleteLater()

    def test_every_model_sees_generated_reference_not_original_inpaint(self):
        original, generated = batch_fixtures._reference(10), batch_fixtures._reference(200)
        for _name, model in SYMMETRY_INFERENCE_MODEL_OPTIONS:
            with self.subTest(model=model):
                thread = _ReferenceGenerationThread(original, "front view", "image-model", api_key="secret", strength=0.5)
                thread.symmetry_model = model
                decision = SymmetryInferenceResult(symmetric=True, model=model)
                with patch("housemaker.reference_generation.generate_object_reference", return_value=generated), patch(
                    "housemaker.reference_symmetry_inference.infer_reference_symmetry", return_value=decision,
                ) as infer:
                    thread.run()
                self.assertIs(infer.call_args.args[0], generated)
                self.assertEqual(infer.call_args.args[1], model)
                self.assertIs(thread.symmetry_result, decision)
                self.assertIsNone(thread.error_message)
                self.assertEqual(thread._api_key, "")
                thread.deleteLater()

    def test_symmetry_error_preserves_generated_image_and_redacts_credentials(self):
        generated = batch_fixtures._reference(200)
        thread = _ReferenceGenerationThread(batch_fixtures._reference(10), "front", "image", api_key="secret", strength=0.5)
        thread.symmetry_model = SYMMETRY_MODEL_LUNA
        with patch("housemaker.reference_generation.generate_object_reference", return_value=generated), patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry", side_effect=RuntimeError("bad secret"),
        ):
            thread.run()
        self.assertIs(thread.result, generated)
        self.assertIsNone(thread.error_message)
        self.assertEqual(thread.symmetry_error, "bad [redacted]")
        thread.deleteLater()


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
