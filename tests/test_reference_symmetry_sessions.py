# ### Imports ###
from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from housemaker import reference_symmetry_inference as symmetry
from housemaker.qwen_ceiling_height_estimation import (
    QwenCeilingHeightCancelled,
    QwenCeilingHeightError,
)
from housemaker.reflect3d_reference_symmetry import SymmetryPrediction


# ### Shared small fixtures ###
def _image() -> np.ndarray:
    return np.full((16, 24, 4), 255, np.uint8)


def _payload(symmetric: bool = True) -> dict:
    return {"symmetric": symmetric, "confidence": 0.9, "reason": "Matching halves."}


def _backend() -> Mock:
    backend = Mock()
    backend.__enter__ = Mock(return_value=backend)
    backend.estimate.return_value = _payload()
    return backend


# ### Lazy local resource reuse and callback isolation ###
class ReferenceSymmetrySessionTests(unittest.TestCase):
    def test_qwen_opens_only_on_first_inference_and_reuses_until_closed(self) -> None:
        backend = _backend()
        with patch(
            "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
            return_value=backend,
        ) as factory:
            session = symmetry.open_reference_symmetry_session(symmetry.SYMMETRY_MODEL_QWEN)
            factory.assert_not_called()
            with session:
                factory.assert_not_called()
                first = session.infer(_image())
                second = session.infer(_image())
                backend.close.assert_not_called()
            session.close()
        factory.assert_called_once()
        backend.__enter__.assert_called_once()
        self.assertEqual(backend.estimate.call_count, 2)
        backend.close.assert_called_once()
        self.assertTrue(first.symmetric)
        self.assertTrue(second.symmetric)
        self.assertGreaterEqual(second.elapsed_seconds, 0)

    def test_qwen_reuse_is_valid_across_sequential_worker_threads(self) -> None:
        backend = _backend()
        results = []
        failures = []
        with patch(
            "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
            return_value=backend,
        ) as factory, symmetry.open_reference_symmetry_session(
            symmetry.SYMMETRY_MODEL_QWEN
        ) as session:
            def infer() -> None:
                try:
                    results.append(session.infer(_image()))
                except Exception as error:  # noqa: BLE001 - transfer worker failures.
                    failures.append(error)

            for _ in range(2):
                worker = threading.Thread(target=infer)
                worker.start()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
        self.assertFalse(failures)
        self.assertEqual(len(results), 2)
        factory.assert_called_once()
        backend.close.assert_called_once()

    def test_finished_worker_cancellation_and_progress_are_not_retained(self) -> None:
        backend = _backend()
        previous_cancelled = False
        first_progress = Mock()
        second_progress = Mock()
        with patch(
            "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
            return_value=backend,
        ), symmetry.open_reference_symmetry_session(symmetry.SYMMETRY_MODEL_QWEN) as session:
            session.infer(
                _image(),
                cancellation_check=lambda: previous_cancelled,
                progress_callback=first_progress,
            )
            previous_cancelled = True
            result = session.infer(_image(), progress_callback=second_progress)
        self.assertTrue(result.symmetric)
        first_progress.assert_called_once()
        second_progress.assert_called_once()
        self.assertEqual(backend.estimate.call_count, 2)

    def test_default_cancellation_and_progress_apply_when_no_override_is_given(self) -> None:
        cancelled = False
        backend = _backend()
        progress = Mock()
        with patch(
            "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
            return_value=backend,
        ), symmetry.open_reference_symmetry_session(
            symmetry.SYMMETRY_MODEL_QWEN,
            cancellation_check=lambda: cancelled,
            progress_callback=progress,
        ) as session:
            session.infer(_image())
            cancelled = True
            with self.assertRaises(symmetry.SymmetryInferenceCancelled):
                session.infer(_image())
            backend.close.assert_called_once()
        progress.assert_called_once()
        self.assertEqual(backend.estimate.call_count, 1)

    def test_closed_session_cannot_restart_a_model(self) -> None:
        session = symmetry.open_reference_symmetry_session(symmetry.SYMMETRY_MODEL_QWEN)
        session.close()
        with patch(
            "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer"
        ) as factory:
            with self.assertRaisesRegex(symmetry.SymmetryInferenceError, "closed"):
                session.infer(_image())
            with self.assertRaisesRegex(symmetry.SymmetryInferenceError, "closed"):
                session.__enter__()
        factory.assert_not_called()

    def test_invalid_image_never_starts_qwen(self) -> None:
        with patch(
            "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer"
        ) as factory, symmetry.open_reference_symmetry_session(
            symmetry.SYMMETRY_MODEL_QWEN
        ) as session, self.assertRaises(symmetry.SymmetryInferenceError):
            session.infer(np.zeros((0, 0, 4), np.uint8))
        factory.assert_not_called()


# ### Backend failures, cancellation and phase teardown ###
class ReferenceSymmetrySessionFailureTests(unittest.TestCase):
    def test_qwen_backend_errors_close_and_preserve_safe_reason(self) -> None:
        for error in (
            QwenCeilingHeightError("The local weights are unavailable."),
            QwenCeilingHeightCancelled("Cancelled."),
            RuntimeError("sk-secret data:image/png;base64,PRIVATE"),
        ):
            backend = _backend()
            backend.estimate.side_effect = error
            with self.subTest(error=error), patch(
                "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
                return_value=backend,
            ), symmetry.open_reference_symmetry_session(symmetry.SYMMETRY_MODEL_QWEN) as session:
                with self.assertRaises(symmetry.SymmetryInferenceError) as caught:
                    session.infer(_image())
                backend.close.assert_called_once()
                if isinstance(error, QwenCeilingHeightCancelled):
                    self.assertIsInstance(caught.exception, symmetry.SymmetryInferenceCancelled)
                if type(error) is QwenCeilingHeightError:
                    self.assertIn("weights are unavailable", str(caught.exception))
                self.assertNotIn("PRIVATE", str(caught.exception))
                self.assertNotIn("sk-secret", str(caught.exception))

    def test_failed_startup_still_closes_partially_opened_backend(self) -> None:
        backend = _backend()
        backend.__enter__.side_effect = QwenCeilingHeightError("Could not load weights.")
        with (
            patch(
                "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
                return_value=backend,
            ),
            symmetry.open_reference_symmetry_session(symmetry.SYMMETRY_MODEL_QWEN) as session,
            self.assertRaisesRegex(symmetry.SymmetryInferenceError, "load weights"),
        ):
            session.infer(_image())
        backend.close.assert_called_once()

    def test_next_request_can_restart_after_a_backend_failure(self) -> None:
        failed = _backend()
        failed.estimate.side_effect = QwenCeilingHeightError("Worker stopped.")
        healthy = _backend()
        with patch(
            "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
            side_effect=[failed, healthy],
        ) as factory, symmetry.open_reference_symmetry_session(
            symmetry.SYMMETRY_MODEL_QWEN
        ) as session:
            with self.assertRaises(symmetry.SymmetryInferenceError):
                session.infer(_image())
            self.assertTrue(session.infer(_image()).symmetric)
        self.assertEqual(factory.call_count, 2)
        failed.close.assert_called_once()
        healthy.close.assert_called_once()

    def test_context_exception_always_releases_local_backend(self) -> None:
        backend = _backend()
        with (
            patch(
                "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer",
                return_value=backend,
            ),
            self.assertRaisesRegex(ValueError, "phase failed"),
            symmetry.open_reference_symmetry_session(symmetry.SYMMETRY_MODEL_QWEN) as session,
        ):
            session.infer(_image())
            raise ValueError("phase failed")
        backend.close.assert_called_once()


# ### Reflect3D model reuse and remote request boundaries ###
class ReferenceSymmetrySessionBackendTests(unittest.TestCase):
    def test_reflect3d_worker_is_reused_and_preserves_one_shot_decision_policy(self) -> None:
        backend = _backend()
        backend.predict.side_effect = [
            SymmetryPrediction(True, 0.9, (1.0, 0.0, 0.0)),
            SymmetryPrediction(False, 0.2, (1.0, 0.0, 0.0)),
        ]
        with patch(
            "housemaker.reflect3d_reference_symmetry.Reflect3DSymmetrySession",
            return_value=backend,
        ) as factory, symmetry.open_reference_symmetry_session(
            symmetry.SYMMETRY_MODEL_REFLECT3D
        ) as session:
            factory.assert_not_called()
            first = session.infer(_image())
            second = session.infer(_image())
        self.assertTrue(first.symmetric)
        self.assertFalse(second.symmetric)
        self.assertIn("below 0.50", second.reason)
        self.assertEqual(backend.predict.call_count, 2)
        factory.assert_called_once()
        backend.close.assert_called_once()

    def test_fully_transparent_reflect_image_does_not_load_model(self) -> None:
        with patch(
            "housemaker.reflect3d_reference_symmetry.Reflect3DSymmetrySession"
        ) as factory, symmetry.open_reference_symmetry_session(
            symmetry.SYMMETRY_MODEL_REFLECT3D
        ) as session, self.assertRaisesRegex(symmetry.SymmetryInferenceError, "transparent"):
            session.infer(np.zeros((8, 8, 4), np.uint8))
        factory.assert_not_called()

    def test_luna_still_sends_individual_requests_without_local_model_load(self) -> None:
        with patch.object(symmetry, "_infer_with_luna", return_value=_payload()) as infer, patch(
            "housemaker.qwen_ceiling_height_estimation.QwenCeilingHeightServer"
        ) as qwen, symmetry.open_reference_symmetry_session(
            symmetry.SYMMETRY_MODEL_LUNA, api_key="sk-test"
        ) as session:
            session.infer(_image())
            session.infer(_image())
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(infer.call_args.kwargs["api_key"], "sk-test")
        qwen.assert_not_called()
        self.assertEqual(session._api_key, "")


# ### Standalone test runner ###
if __name__ == "__main__":
    unittest.main()
