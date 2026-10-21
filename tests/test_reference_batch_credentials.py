"""Reference providers and symmetry inference keep their credentials separate."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from housemaker.openai_batch import LUNA_BATCH_MODEL, BatchResult
from housemaker.reference_generation_batch import (
    APIReferenceBatchThread,
    LocalReferenceBatchItem,
    LocalReferenceBatchThread,
    OpenAIReferenceBatchThread,
)
from housemaker.reference_symmetry_inference import (
    SYMMETRY_MODEL_LUNA,
    SymmetryInferenceError,
    SymmetryInferenceResult,
)

# ### Offline fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)
IMAGE_KEY = "meshy-secret"
SYMMETRY_KEY = "openai-secret"
MESHY_REFERENCE_MODEL = "meshy-image-to-image"


class ReferenceBatchCredentialsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((12, 12, 4), 255, np.uint8)
        self.items = (LocalReferenceBatchItem("object", self.image),)
        self.outcomes = {}
        self.workers = []

    def tearDown(self) -> None:
        for worker in self.workers:
            worker.deleteLater()
        APP.processEvents()

    def _worker(self, worker_type=APIReferenceBatchThread, **kwargs):
        options = {
            "api_key": IMAGE_KEY, "symmetry_api_key": SYMMETRY_KEY,
            "prompt": "Show the object", "strength": 0.5,
        }
        options.update(kwargs)
        worker = worker_type(
            self.items, MESHY_REFERENCE_MODEL,
            options.pop("symmetry_model", SYMMETRY_MODEL_LUNA), **options,
        )
        worker.item_finished.connect(
            lambda job_id, result, error, cancelled: self.outcomes.update(
                {job_id: (result, error, cancelled)},
            ),
        )
        self.workers.append(worker)
        return worker

    def _assert_keys_cleared(self, worker) -> None:
        self.assertEqual(worker._api_key, "")
        self.assertEqual(worker._symmetry_api_key, "")

    def _assert_redacted_error(self, worker) -> None:
        message = self.outcomes["object"][1]
        self.assertNotIn(IMAGE_KEY, message)
        self.assertNotIn(SYMMETRY_KEY, message)
        self.assertIn("[redacted]", message)
        self._assert_keys_cleared(worker)

    # ### Credential routing ###
    def test_meshy_images_use_image_key_and_regular_symmetry_uses_openai_key(self) -> None:
        worker = self._worker()
        session = Mock()
        session.infer.return_value = SymmetryInferenceResult(True, SYMMETRY_MODEL_LUNA)

        @contextmanager
        def symmetry_session(model, **kwargs):
            self.assertEqual(model, SYMMETRY_MODEL_LUNA)
            self.assertEqual(kwargs["api_key"], SYMMETRY_KEY)
            yield session

        with patch(
            "housemaker.reference_generation.generate_object_reference", return_value=self.image,
        ) as generate, patch(
            "housemaker.reference_symmetry_inference.open_reference_symmetry_session",
            side_effect=symmetry_session,
        ):
            worker.run()

        self.assertEqual(generate.call_args.kwargs["model"], MESHY_REFERENCE_MODEL)
        self.assertEqual(generate.call_args.kwargs["api_key"], IMAGE_KEY)
        self.assertTrue(self.outcomes["object"][0].symmetric)
        self._assert_keys_cleared(worker)

    def test_meshy_images_and_luna_batch_symmetry_use_separate_keys(self) -> None:
        worker = self._worker(symmetry_model=LUNA_BATCH_MODEL)
        decision = SymmetryInferenceResult(True, LUNA_BATCH_MODEL)
        with patch(
            "housemaker.reference_generation.generate_object_reference", return_value=self.image,
        ) as generate, patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry_batch",
            return_value={"object": decision},
        ) as infer:
            worker.run()

        self.assertEqual(generate.call_args.kwargs["api_key"], IMAGE_KEY)
        self.assertEqual(infer.call_args.kwargs["api_key"], SYMMETRY_KEY)
        self.assertIs(self.outcomes["object"][0], decision)
        self._assert_keys_cleared(worker)

    def test_omitted_symmetry_key_preserves_legacy_api_key_fallback(self) -> None:
        worker = self._worker(symmetry_api_key=None, symmetry_model=LUNA_BATCH_MODEL)
        with patch(
            "housemaker.reference_generation.generate_object_reference", return_value=self.image,
        ), patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry_batch",
            return_value={"object": SymmetryInferenceResult(True, LUNA_BATCH_MODEL)},
        ) as infer:
            worker.run()
        self.assertEqual(infer.call_args.kwargs["api_key"], IMAGE_KEY)
        self._assert_keys_cleared(worker)

    def test_explicit_empty_symmetry_key_does_not_use_meshy_key(self) -> None:
        worker = self._worker(symmetry_api_key="", symmetry_model=LUNA_BATCH_MODEL)
        with patch(
            "housemaker.reference_generation.generate_object_reference", return_value=self.image,
        ), patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry_batch",
            return_value={"object": SymmetryInferenceResult(True, LUNA_BATCH_MODEL)},
        ) as infer:
            worker.run()
        self.assertEqual(infer.call_args.kwargs["api_key"], "")
        self._assert_keys_cleared(worker)

    def test_local_images_keep_separate_remote_symmetry_credentials(self) -> None:
        worker = self._worker(LocalReferenceBatchThread, symmetry_model=LUNA_BATCH_MODEL)

        @contextmanager
        def image_session(_model, **_kwargs):
            yield Mock(return_value=self.image)

        with patch(
            "housemaker.reference_generation.open_object_reference_session",
            side_effect=image_session,
        ), patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry_batch",
            return_value={"object": SymmetryInferenceResult(True, LUNA_BATCH_MODEL)},
        ) as infer:
            worker.run()
        self.assertEqual(infer.call_args.kwargs["api_key"], SYMMETRY_KEY)
        self._assert_keys_cleared(worker)

    def test_openai_image_batch_and_symmetry_batch_use_their_respective_keys(self) -> None:
        worker = self._worker(OpenAIReferenceBatchThread, symmetry_model=LUNA_BATCH_MODEL)
        worker.model = LUNA_BATCH_MODEL
        with patch(
            "housemaker.reference_generation_batch.run_responses_batch",
            return_value={"object": BatchResult({"result": "mock"}, None)},
        ) as generate, patch(
            "housemaker.plan_image_correction._decode_openai_image_generation_result",
            return_value=b"mock image",
        ), patch(
            "housemaker.reference_generation._decode_generated_reference", return_value=self.image,
        ), patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry_batch",
            return_value={"object": SymmetryInferenceResult(True, LUNA_BATCH_MODEL)},
        ) as infer:
            worker.run()

        self.assertEqual(generate.call_args.kwargs["api_key"], IMAGE_KEY)
        self.assertEqual(infer.call_args.kwargs["api_key"], SYMMETRY_KEY)
        self._assert_keys_cleared(worker)

    # ### Error safety and cleanup ###
    def test_image_provider_errors_redact_both_keys(self) -> None:
        worker = self._worker()
        with patch(
            "housemaker.reference_generation.generate_object_reference",
            side_effect=RuntimeError(f"Image request {IMAGE_KEY}; symmetry {SYMMETRY_KEY}"),
        ):
            worker.run()
        self._assert_redacted_error(worker)

    def test_returned_symmetry_batch_errors_redact_both_keys(self) -> None:
        worker = self._worker(symmetry_model=LUNA_BATCH_MODEL)
        with patch(
            "housemaker.reference_generation.generate_object_reference", return_value=self.image,
        ), patch(
            "housemaker.reference_symmetry_inference.infer_reference_symmetry_batch",
            return_value={"object": SymmetryInferenceError(f"{IMAGE_KEY} {SYMMETRY_KEY}")},
        ):
            worker.run()
        self._assert_redacted_error(worker)

    def test_symmetry_setup_errors_redact_both_keys_and_clear_credentials(self) -> None:
        worker = self._worker()
        with patch(
            "housemaker.reference_generation.generate_object_reference", return_value=self.image,
        ), patch(
            "housemaker.reference_symmetry_inference.open_reference_symmetry_session",
            side_effect=RuntimeError(f"{IMAGE_KEY} {SYMMETRY_KEY}"),
        ):
            worker.run()
        self._assert_redacted_error(worker)
        self.assertNotIn(SYMMETRY_KEY, worker.error_message)

    def test_openai_image_batch_response_errors_redact_both_keys(self) -> None:
        worker = self._worker(OpenAIReferenceBatchThread)
        worker.model = LUNA_BATCH_MODEL
        with patch(
            "housemaker.reference_generation_batch.run_responses_batch",
            return_value={"object": BatchResult(None, f"{IMAGE_KEY} {SYMMETRY_KEY}")},
        ):
            worker.run()
        self._assert_redacted_error(worker)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
