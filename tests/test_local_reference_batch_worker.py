"""Resident local sessions are phased, reusable, cancellable, and isolated."""

# ### Environment and imports ###
from __future__ import annotations

import os
import threading
import unittest
from contextlib import contextmanager
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QApplication

from housemaker.plan_correction_models import PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1
from housemaker.reference_generation_batch import (
    LocalReferenceBatchItem,
    LocalReferenceBatchThread,
)
from housemaker.reference_symmetry_inference import (
    SYMMETRY_MODEL_QWEN,
    SymmetryInferenceResult,
)

# ### Controlled resident-session fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _image(value: int) -> np.ndarray:
    image = np.full((8, 10, 4), value, dtype=np.uint8)
    image[:, :, 3] = 255
    return image


class _ResidentSessions:
    """Record model lifetimes without importing native model backends."""

    def __init__(self) -> None:
        self.events = []
        self.render_effect = None
        self.infer_effect = None
        self.image_open_error = None
        self.image_close_error = None
        self.symmetry_open_error = None
        self.image_arguments = None
        self.symmetry_arguments = None

    @contextmanager
    def images(self, model, **kwargs):
        self.image_arguments = model, kwargs
        self.events.append("image open")
        if self.image_open_error:
            raise self.image_open_error
        try:
            yield self.generate
        finally:
            self.events.append("image close")
            if self.image_close_error:
                raise self.image_close_error

    def generate(self, image, prompt, **kwargs):
        value = int(image[0, 0, 0])
        self.events.append(("render", value))
        kwargs["progress_callback"]("Rendering", 50)
        if self.render_effect is not None:
            self.render_effect(value, kwargs)
        return _image(value + 100)

    @contextmanager
    def symmetry(self, model, **kwargs):
        self.symmetry_arguments = model, kwargs
        self.events.append("symmetry open")
        if self.symmetry_open_error:
            raise self.symmetry_open_error
        try:
            yield self
        finally:
            self.events.append("symmetry close")

    def infer(self, image, **kwargs):
        value = int(image[0, 0, 0])
        self.events.append(("infer", value))
        kwargs["progress_callback"]("Inferring")
        if self.infer_effect is not None:
            self.infer_effect(value, kwargs)
        return SymmetryInferenceResult(
            symmetric=value != 120, model=SYMMETRY_MODEL_QWEN,
            confidence=0.9,
        )


class _MainThreadReceiver(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    @Slot(str, object)
    def reference_ready(self, job_id, image) -> None:
        self.calls.append((job_id, threading.get_ident(), int(image[0, 0, 0])))


# ### Worker lifecycle and phase tests ###
class LocalReferenceBatchWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sessions = _ResidentSessions()
        self.thread = LocalReferenceBatchThread(
            tuple(LocalReferenceBatchItem(name, _image(value)) for name, value in (
                ("one", 10), ("two", 20), ("three", 30),
            )),
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1, SYMMETRY_MODEL_QWEN,
            api_key="secret", prompt="Show the object from the front",
            strength=0.5, negative_prompt="blur",
        )
        self.ready = []
        self.finished = []
        self.progress = []
        self.thread.reference_ready.connect(lambda job_id, image: self.ready.append((job_id, image)))
        self.thread.item_finished.connect(lambda *args: self.finished.append(args))
        self.thread.item_progress.connect(lambda *args: self.progress.append(args))
        self.image_patch = patch(
            "housemaker.reference_generation.open_object_reference_session", self.sessions.images,
        )
        self.symmetry_patch = patch(
            "housemaker.reference_symmetry_inference.open_reference_symmetry_session", self.sessions.symmetry,
        )
        self.image_patch.start()
        self.symmetry_patch.start()

    def tearDown(self) -> None:
        self.thread.wait(10_000)
        self.thread.deleteLater()
        self.symmetry_patch.stop()
        self.image_patch.stop()
        APP.processEvents()

    def _outcomes(self):
        self.assertEqual(len(self.finished), 3)
        self.assertEqual(len({job_id for job_id, *_rest in self.finished}), 3)
        return {job_id: (decision, error, cancelled) for job_id, decision, error, cancelled in self.finished}

    def test_one_model_load_per_phase_with_all_images_before_symmetry(self) -> None:
        self.thread.run()
        self.assertEqual(self.sessions.events, [
            "image open", ("render", 10), ("render", 20), ("render", 30),
            "image close", "symmetry open", ("infer", 110), ("infer", 120),
            ("infer", 130), "symmetry close",
        ])
        self.assertEqual([job_id for job_id, _image in self.ready], ["one", "two", "three"])
        outcomes = self._outcomes()
        self.assertEqual([outcomes[key][0].symmetric for key in ("one", "two", "three")], [True, False, True])
        self.assertTrue(all(error == "" and not cancelled for _decision, error, cancelled in outcomes.values()))
        self.assertEqual(self.sessions.image_arguments[0], PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1)
        self.assertEqual(self.sessions.symmetry_arguments[0], SYMMETRY_MODEL_QWEN)
        self.assertEqual(self.thread._api_key, "")

    def test_disabled_inference_accepts_references_without_opening_symmetry_model(self) -> None:
        self.thread.symmetry_model = None
        self.thread.run()
        self.assertEqual(self.sessions.events, [
            "image open", ("render", 10), ("render", 20), ("render", 30), "image close",
        ])
        self.assertEqual([job_id for job_id, _image in self.ready], ["one", "two", "three"])
        self.assertIsNone(self.sessions.symmetry_arguments)
        self.assertTrue(all(outcome == (None, "", False) for outcome in self._outcomes().values()))
        self.assertEqual(self.thread._api_key, "")

    def test_disabled_inference_preserves_individual_cancellation(self) -> None:
        self.thread.symmetry_model = None
        self.thread.cancel_item("two")
        self.thread.run()
        self.assertEqual([job_id for job_id, _image in self.ready], ["one", "three"])
        self.assertNotIn("symmetry open", self.sessions.events)
        outcomes = self._outcomes()
        self.assertEqual(outcomes["two"], (None, "", True))
        self.assertEqual(outcomes["one"], (None, "", False))
        self.assertEqual(outcomes["three"], (None, "", False))

    def test_render_error_is_redacted_and_only_that_item_skips_symmetry(self) -> None:
        def render(value, _kwargs):
            if value == 20:
                raise RuntimeError("render failed with secret")

        self.sessions.render_effect = render
        self.thread.run()
        self.assertEqual([job_id for job_id, _image in self.ready], ["one", "three"])
        self.assertNotIn(("infer", 120), self.sessions.events)
        outcomes = self._outcomes()
        self.assertEqual(outcomes["two"], (None, "render failed with [redacted]", False))
        self.assertTrue(outcomes["one"][0].symmetric)
        self.assertTrue(outcomes["three"][0].symmetric)
        self.assertEqual(self.sessions.events[-1], "symmetry close")

    def test_symmetry_error_keeps_all_images_and_continues_with_the_same_session(self) -> None:
        def infer(value, _kwargs):
            if value == 120:
                raise RuntimeError("symmetry failed with secret")

        self.sessions.infer_effect = infer
        self.thread.run()
        self.assertEqual(len(self.ready), 3)
        outcomes = self._outcomes()
        self.assertEqual(outcomes["two"], (None, "Symmetry inference failed: symmetry failed with [redacted]", False))
        self.assertIsNotNone(outcomes["three"][0])
        self.assertEqual(self.sessions.events.count("symmetry open"), 1)
        self.assertEqual(self.sessions.events[-1], "symmetry close")

    def test_queued_item_cancellation_does_not_interrupt_other_model_requests(self) -> None:
        self.thread.cancel_item("two")
        self.thread.run()
        self.assertNotIn(("render", 20), self.sessions.events)
        self.assertNotIn(("infer", 120), self.sessions.events)
        outcomes = self._outcomes()
        self.assertEqual(outcomes["two"], (None, "", True))
        self.assertIsNotNone(outcomes["one"][0])
        self.assertIsNotNone(outcomes["three"][0])

    def test_current_render_cancellation_drops_only_that_result(self) -> None:
        def render(value, kwargs):
            if value == 10:
                self.thread.cancel_item("one")
                self.assertTrue(kwargs["cancellation_check"]())

        self.sessions.render_effect = render
        self.thread.run()
        self.assertEqual([job_id for job_id, _image in self.ready], ["two", "three"])
        self.assertEqual(self._outcomes()["one"], (None, "", True))
        self.assertEqual(self.sessions.events.count("image open"), 1)

    def test_cancellation_between_phases_skips_only_affected_symmetry(self) -> None:
        self.thread.reference_ready.connect(
            lambda job_id, _image: self.thread.cancel_item(job_id) if job_id == "one" else None,
        )
        self.thread.run()
        self.assertEqual(len(self.ready), 3)
        self.assertNotIn(("infer", 110), self.sessions.events)
        self.assertEqual(self._outcomes()["one"], (None, "", True))

    def test_cancel_all_during_render_closes_image_session_without_opening_symmetry(self) -> None:
        interrupted = [False]

        def render(value, kwargs):
            if value == 20:
                interrupted[0] = True
                self.assertTrue(kwargs["cancellation_check"]())

        self.sessions.render_effect = render
        with patch.object(self.thread, "isInterruptionRequested", side_effect=lambda: interrupted[0]):
            self.thread.run()
        self.assertEqual([job_id for job_id, _image in self.ready], ["one"])
        self.assertNotIn(("render", 30), self.sessions.events)
        self.assertEqual(self.sessions.events[-1], "image close")
        self.assertNotIn("symmetry open", self.sessions.events)
        self.assertTrue(all(cancelled for _decision, _error, cancelled in self._outcomes().values()))
        self.assertEqual(self.thread._api_key, "")

    def test_cancel_all_during_symmetry_still_closes_the_resident_model(self) -> None:
        interrupted = [False]

        def infer(value, kwargs):
            if value == 110:
                interrupted[0] = True
                self.assertTrue(kwargs["cancellation_check"]())

        self.sessions.infer_effect = infer
        with patch.object(self.thread, "isInterruptionRequested", side_effect=lambda: interrupted[0]):
            self.thread.run()
        self.assertEqual(len(self.ready), 3)
        self.assertNotIn(("infer", 120), self.sessions.events)
        self.assertEqual(self.sessions.events[-1], "symmetry close")
        self.assertTrue(all(cancelled for _decision, _error, cancelled in self._outcomes().values()))

    def test_image_model_setup_failure_finishes_every_item_once(self) -> None:
        self.sessions.image_open_error = RuntimeError("image model secret unavailable")
        self.thread.run()
        self.assertFalse(self.ready)
        self.assertEqual(self.sessions.events, ["image open"])
        self.assertTrue(all(error == "image model [redacted] unavailable" for _decision, error, _cancelled in self._outcomes().values()))
        self.assertEqual(self.thread._api_key, "")

    def test_image_cleanup_failure_keeps_ready_images_but_never_overlaps_models(self) -> None:
        self.sessions.image_close_error = RuntimeError("image process did not close")
        self.thread.run()
        self.assertEqual(len(self.ready), 3)
        self.assertEqual(self.sessions.events[-1], "image close")
        self.assertNotIn("symmetry open", self.sessions.events)
        self.assertTrue(all(error == "image process did not close" for _decision, error, _cancelled in self._outcomes().values()))

    def test_symmetry_model_setup_failure_preserves_all_completed_references(self) -> None:
        self.sessions.symmetry_open_error = RuntimeError("symmetry model secret unavailable")
        self.thread.run()
        self.assertEqual(len(self.ready), 3)
        self.assertLess(self.sessions.events.index("image close"), self.sessions.events.index("symmetry open"))
        self.assertTrue(all(error == "symmetry model [redacted] unavailable" for _decision, error, _cancelled in self._outcomes().values()))

    def test_real_qthread_delivers_reference_signals_to_gui_receiver_on_main_thread(self) -> None:
        receiver = _MainThreadReceiver()
        self.thread.reference_ready.connect(receiver.reference_ready)
        main_thread_id = threading.get_ident()
        self.thread.start()
        self.assertTrue(self.thread.wait(10_000))
        APP.processEvents()
        self.assertEqual(receiver.calls, [
            ("one", main_thread_id, 110), ("two", main_thread_id, 120),
            ("three", main_thread_id, 130),
        ])
        self.assertEqual(self.sessions.events[-1], "symmetry close")
        receiver.deleteLater()


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
