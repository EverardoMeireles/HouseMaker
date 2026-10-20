"""Model-free tests for the reproducible local symmetry benchmark."""

# ### Imports ###
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from housemaker.reference_symmetry_inference import SymmetryInferenceResult

# ### Benchmark module loading ###
SPEC = importlib.util.spec_from_file_location(
    "symmetry_benchmark",
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark_reference_symmetry.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


# ### Input and scheduling checks ###
class BenchmarkReferenceSymmetryTests(unittest.TestCase):
    def test_inputs_are_decoded_without_resizing_or_color_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory, f"object-{index}.png") for index in range(2)]
            image = np.full((45, 100, 4), (12, 34, 56, 255), dtype=np.uint8)
            for path in paths:
                cv2.imwrite(str(path), image)
            inputs = benchmark.load_inputs(paths)
            self.assertEqual(len(inputs), 2)
            self.assertEqual(inputs[0][0]["width"], 100)
            self.assertEqual(inputs[0][0]["sha256"], inputs[1][0]["sha256"])
            np.testing.assert_array_equal(inputs[0][1], image)

    def test_two_inputs_required_for_cold_and_warm_comparison(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            benchmark.load_inputs([])

    def test_prepared_inputs_use_the_application_white_reference_canvas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "object.png")
            image = np.full((45, 100, 4), (12, 34, 56, 255), dtype=np.uint8)
            cv2.imwrite(str(path), image)
            inputs = benchmark.load_inputs([path, path], prepare_reference=True)
        self.assertEqual(inputs[0][1].shape, (1024, 1024, 4))
        self.assertTrue(inputs[0][0]["prepared_reference"])
        np.testing.assert_array_equal(inputs[0][1][0, 0], (255, 255, 255, 255))

    def test_paid_models_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only local"):
            benchmark.benchmark_model(
                "gpt-5.6-luna", [], call_timeout=1, session_timeout=2
            )

    def test_one_session_is_reused_and_closed(self):
        sessions = []

        class FakeSession:
            def __init__(self, model, **_kwargs):
                self.model, self.calls, self.closed = model, 0, False
                sessions.append(self)

            def __enter__(self):
                return self

            def infer(self, image, **kwargs):
                self.calls += 1
                assert image.shape == (2, 2, 3)
                assert not kwargs["cancellation_check"]()
                return SymmetryInferenceResult(True, self.model)

            def close(self):
                self.closed = True

        inputs = [
            ({"path": str(i), "sha256": str(i)}, np.zeros((2, 2, 3), np.uint8))
            for i in range(3)
        ]
        with (
            patch.object(benchmark, "gpu_snapshot", return_value=[]),
            patch("builtins.print"),
        ):
            report = benchmark.benchmark_model(
                benchmark.LOCAL_MODELS[0],
                inputs,
                call_timeout=1,
                session_timeout=2,
                session_factory=FakeSession,
            )
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].calls, 3)
        self.assertTrue(sessions[0].closed)
        self.assertEqual(report["completed_requests"], 3)
        self.assertIsNotNone(report["cold_first_request_seconds"])
        self.assertIsNotNone(report["warm_median_seconds"])

    def test_failure_is_reported_without_fallback(self):
        class FailingSession:
            def __init__(self, *_args, **_kwargs):
                self.closed = False

            def __enter__(self):
                return self

            def infer(self, *_args, **_kwargs):
                raise RuntimeError("Test inference failure")

            def close(self):
                self.closed = True

        inputs = [({"path": "1", "sha256": "1"}, np.zeros((2, 2, 3), np.uint8))] * 2
        with (
            patch.object(benchmark, "gpu_snapshot", return_value=[]),
            patch("builtins.print"),
        ):
            report = benchmark.benchmark_model(
                benchmark.LOCAL_MODELS[0],
                inputs,
                call_timeout=1,
                session_timeout=2,
                session_factory=FailingSession,
            )
        self.assertEqual(len(report["requests"]), 1)
        self.assertIn("Test inference failure", report["requests"][0]["error"])
        self.assertEqual(report["completed_requests"], 0)
        self.assertIsNone(report["warm_median_seconds"])


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
