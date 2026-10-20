"""Grouped cloud reference jobs without paid requests or local inference."""

# ### Environment and imports ###
from __future__ import annotations

import base64
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import test_local_reference_batch_workspace as workspace_fixtures
from PySide6.QtWidgets import QApplication

from housemaker.openai_batch import (
    LUNA_BATCH_MODEL,
    LUNA_MODEL,
    BatchResult,
    OpenAIBatchCancelled,
)
from housemaker.reference_generation_batch import (
    APIReferenceBatchThread,
    LocalReferenceBatchItem,
    OpenAIReferenceBatchThread,
)
from housemaker.reference_symmetry_inference import (
    SymmetryInferenceError,
    SymmetryInferenceResult,
)

# ### Offline response fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _image(value: int) -> np.ndarray:
    pixels = np.full((12, 14, 4), value, np.uint8)
    pixels[:, :, 3] = 255
    return pixels


def _image_response(value: int) -> dict:
    success, png = cv2.imencode(".png", _image(value))
    assert success
    return {
        "status": "completed",
        "output": [{
            "type": "image_generation_call", "status": "completed",
            "result": base64.b64encode(png).decode("ascii"),
        }],
    }


# ### Grouped worker behavior ###
class OpenAIReferenceBatchWorkerTests(unittest.TestCase):
    def setUp(self):
        self.worker = OpenAIReferenceBatchThread(
            tuple(LocalReferenceBatchItem(name, _image(value), f"Reference {name}")
                  for name, value in (("one", 10), ("two", 20), ("three", 30))),
            LUNA_BATCH_MODEL, None, api_key="secret", prompt="Fallback",
            strength=0.5,
        )
        self.ready = {}
        self.finished = {}
        self.progress = []
        self.worker.reference_ready.connect(lambda key, image: self.ready.update({key: image}))
        self.worker.item_finished.connect(
            lambda key, result, error, cancelled: self.finished.update({key: (result, error, cancelled)})
        )
        self.worker.item_progress.connect(lambda *args: self.progress.append(args))

    def tearDown(self):
        self.worker.deleteLater()
        APP.processEvents()

    def _responses(self, requests, **kwargs):
        self.assertEqual(kwargs["api_key"], "secret")
        kwargs["progress_callback"]("Batch queued (up to 24 hours)")
        return {
            key: BatchResult(_image_response(value), None)
            for key, value in reversed(tuple(zip(requests, (110, 120, 130))))
        }

    def _run(self, side_effect=None):
        with patch(
            "housemaker.reference_generation_batch.run_responses_batch",
            side_effect=self._responses if side_effect is None else side_effect,
        ) as submit:
            self.worker.run()
        return submit

    def test_one_batch_contains_all_object_prompts_and_maps_unordered_images(self):
        submit = self._run()
        submit.assert_called_once()
        requests = submit.call_args.args[0]
        self.assertEqual(tuple(requests), ("one", "two", "three"))
        for key, body in requests.items():
            self.assertEqual(body["model"], LUNA_MODEL)
            self.assertIn(f"Reference {key}", body["input"][0]["content"][0]["text"])
            tool = body["tools"][0]
            self.assertEqual((tool["action"], tool["background"]), ("generate", "opaque"))
        for key, value in (("one", 110), ("two", 120), ("three", 130)):
            self.assertEqual(int(self.ready[key][512, 512, 0]), value)
            self.assertEqual(self.finished[key], (None, "", False))
        self.assertTrue(any("queued" in stage for _key, stage, _percent in self.progress))
        self.assertEqual(self.worker._api_key, "")

    def test_partial_failure_keeps_successful_references(self):
        def response(requests, **kwargs):
            outcomes = self._responses(requests, **kwargs)
            outcomes["two"] = BatchResult(None, "Model request rejected")
            return outcomes

        self._run(response)
        self.assertEqual(set(self.ready), {"one", "three"})
        self.assertIn("Model request rejected", self.finished["two"][1])
        self.assertEqual(self.finished["one"], (None, "", False))

    def test_individual_cancellation_does_not_cancel_other_submitted_requests(self):
        def response(requests, **kwargs):
            self.worker.cancel_item("two")
            self.assertFalse(kwargs["cancellation_check"]())
            return self._responses(requests, **kwargs)

        self._run(response)
        self.assertEqual(set(self.ready), {"one", "three"})
        self.assertTrue(self.finished["two"][2])
        self.assertFalse(self.finished["one"][2])

    def test_pre_cancelled_item_is_not_submitted(self):
        self.worker.cancel_item("two")
        submit = self._run()
        self.assertEqual(tuple(submit.call_args.args[0]), ("one", "three"))
        self.assertTrue(self.finished["two"][2])

    def test_cancel_all_interrupts_the_remote_wait_and_applies_no_images(self):
        def response(_requests, **kwargs):
            interrupted = patch.object(self.worker, "isInterruptionRequested", return_value=True)
            interrupted.start()
            self.addCleanup(interrupted.stop)
            self.assertTrue(kwargs["cancellation_check"]())
            raise OpenAIBatchCancelled("Remote cancellation requested")

        self._run(response)
        self.assertFalse(self.ready)
        self.assertTrue(all(item[2] for item in self.finished.values()))

    def test_symmetry_batch_groups_generated_images_and_keeps_references_on_error(self):
        self.worker.symmetry_model = LUNA_BATCH_MODEL
        decisions = {
            "one": SymmetryInferenceResult(True, LUNA_BATCH_MODEL),
            "two": SymmetryInferenceError("No symmetry result"),
            "three": SymmetryInferenceResult(False, LUNA_BATCH_MODEL),
        }
        with patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry_batch",
            return_value=decisions,
        ) as infer:
            self._run()
        infer.assert_called_once()
        self.assertEqual(set(infer.call_args.args[0]), {"one", "two", "three"})
        self.assertEqual(set(self.ready), {"one", "two", "three"})
        self.assertTrue(self.finished["one"][0].symmetric)
        self.assertIn("No symmetry result", self.finished["two"][1])
        self.assertFalse(self.finished["three"][0].symmetric)

    def test_standard_image_provider_finishes_images_before_one_symmetry_batch(self):
        worker = APIReferenceBatchThread(
            self.worker.items, LUNA_MODEL, LUNA_BATCH_MODEL,
            api_key="secret", prompt="Fallback", strength=0.5,
        )
        self.addCleanup(worker.deleteLater)
        events = []

        def generate(image, prompt, **kwargs):
            self.assertEqual(kwargs["model"], LUNA_MODEL)
            self.assertEqual(kwargs["api_key"], "secret")
            events.append(prompt)
            return _image(int(image[0, 0, 0]) + 100)

        def infer(images, **_kwargs):
            self.assertEqual(events, ["Reference one", "Reference two", "Reference three"])
            events.append("one symmetry batch")
            return {key: SymmetryInferenceResult(True, LUNA_BATCH_MODEL) for key in images}

        with patch(
            "housemaker.reference_generation.generate_object_reference", side_effect=generate,
        ), patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry_batch",
            side_effect=infer,
        ) as inference:
            worker.run()
        inference.assert_called_once()
        self.assertEqual(events[-1], "one symmetry batch")


# ### Editor queue integration ###
class OpenAIReferenceBatchWorkspaceTests(unittest.TestCase):
    setUp = workspace_fixtures.LocalReferenceBatchWorkspaceTests.setUp
    tearDown = workspace_fixtures.LocalReferenceBatchWorkspaceTests.tearDown
    _paint = workspace_fixtures.LocalReferenceBatchWorkspaceTests._paint

    def test_batch_model_launches_one_shared_worker_with_individual_prompts(self):
        controlled = workspace_fixtures._ControlledLocalReferenceBatchThread
        self.workspace.set_reference_generation_model(LUNA_BATCH_MODEL)
        self.workspace.set_object_symmetry_inference(False, LUNA_BATCH_MODEL)
        expected = []
        for prompt in ("First object", "Second object", "Third object"):
            self.workspace.reference_generation_prompt_edit.setText(prompt)
            expected.append(prompt)
            self.workspace.next_inpaint_button.click()
        with patch("housemaker.merged_generation_workspace._OpenAIReferenceBatchThread", controlled):
            self.workspace.generate_all_references_button.click()
        self.assertEqual(len(controlled.created), 1)
        worker = controlled.created[0]
        self.assertEqual(worker.model, LUNA_BATCH_MODEL)
        self.assertEqual([item.prompt for item in worker.items], expected)
        self.assertIn("24 hours", self.workspace.reference_generation_status_label.text())
        self.assertFalse(self.workspace.reference_generation_prompt_edit.isEnabled())
        for index in range(3):
            worker.render(index, 100 + index)
            worker.item_finished.emit(worker.items[index].job_id, None, "", False)
        worker.complete()
        self.assertIsNone(self.workspace._local_reference_batch_thread)
        self.assertTrue(self.workspace.reference_generation_prompt_edit.isEnabled())

    def test_batch_symmetry_groups_standard_api_reference_generation(self):
        controlled = workspace_fixtures._ControlledLocalReferenceBatchThread
        self.workspace.set_reference_generation_model(LUNA_MODEL)
        self.workspace.set_object_symmetry_inference(True, LUNA_BATCH_MODEL)
        with patch("housemaker.merged_generation_workspace._APIReferenceBatchThread", controlled):
            self.workspace.generate_all_references_button.click()
        worker = controlled.created[0]
        self.assertEqual(worker.model, LUNA_MODEL)
        self.assertEqual(worker.symmetry_model, LUNA_BATCH_MODEL)
        self.assertEqual(len(worker.items), 3)
        self.assertIn("24 hours", self.workspace.reference_generation_status_label.text())
        for index in range(3):
            worker.render(index, 100 + index)
            worker.decide(index, True)
        worker.complete()


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
