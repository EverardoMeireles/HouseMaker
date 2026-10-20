"""CPU-only protocol, preprocessing, policy, and setup tests for Reflect3D."""

# ### Imports ###
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import numpy as np

from housemaker.reflect3d_reference_setup import (
    REFLECT3D_SOURCE_SHA256,
    REFLECT3D_WEIGHTS_SHA256,
    Reflect3DBundle,
    _download_asset,
    _prepare_runtime,
    _verified_asset,
    ensure_reflect3d_model,
)
from housemaker.reflect3d_reference_symmetry import (
    REFLECT3D_MODEL_ID,
    Reflect3DCancelled,
    Reflect3DError,
    Reflect3DSymmetrySession,
    SymmetryPrediction,
    infer_reflect3d_symmetry,
    prediction_from_planes,
    prepare_reflect3d_pixels,
)

# ### Synthetic object pixels and worker messages ###
CROP = np.full((20, 40, 3), (10, 20, 30), np.uint8)


def _result(identifier=1, *, normals=None, probabilities=None):
    return {"event": "result", "id": identifier, "normals": normals or [[1, 0, 0]],
            "probabilities": probabilities or [0.8], "inference_seconds": 0.025}


class FakeProcess:
    def __init__(self, messages):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("\n".join(json.dumps(item) for item in messages) + "\n")
        self.stderr = io.StringIO()
        self.returncode = None
        self.pid = 424242

    def poll(self):
        return self.returncode


# ### Image preparation and explicit symmetry decisions ###
class Reflect3DPolicyTests(unittest.TestCase):
    def test_model_identity_is_not_a_language_model(self):
        self.assertEqual(REFLECT3D_MODEL_ID, "reflect3d-local")

    def test_preserves_aspect_ratio_with_white_padding_and_converts_bgr_to_rgb(self):
        pixels = prepare_reflect3d_pixels(CROP)
        self.assertEqual(pixels.shape, (224, 224, 3))
        np.testing.assert_array_equal(pixels[112, 112], [30, 20, 10])
        np.testing.assert_array_equal(pixels[0, 112], [255, 255, 255])
        np.testing.assert_array_equal(pixels[112, 0], [255, 255, 255])
        colored = np.any(pixels < 240, axis=2)
        ys, xs = np.nonzero(colored)
        self.assertAlmostEqual((xs.max() - xs.min()) / (ys.max() - ys.min()), 2, delta=0.15)

    def test_genuine_alpha_crops_to_the_object_and_keeps_black_pixels_opaque(self):
        crop = np.zeros((40, 40, 4), np.uint8)
        crop[15:25, 10:30, 3] = 255
        pixels = prepare_reflect3d_pixels(crop)
        np.testing.assert_array_equal(pixels[112, 112], [0, 0, 0])
        np.testing.assert_array_equal(pixels[0, 112], [255, 255, 255])

    def test_invalid_or_fully_transparent_crops_fail_explicitly(self):
        for crop in (np.zeros((0, 3, 3), np.uint8), np.zeros((3, 3), np.uint8),
                     np.zeros((3, 3, 3), np.float32), np.zeros((3, 3, 4), np.uint8)):
            with self.subTest(shape=crop.shape, dtype=crop.dtype), self.assertRaises(Reflect3DError):
                prepare_reflect3d_pixels(crop)

    def test_accepts_high_confidence_object_space_plane_at_a_perspective_angle(self):
        result = prediction_from_planes([[-0.8, 0.1, -0.6]], [0.9])
        self.assertTrue(result.symmetric_division)
        self.assertEqual(result.confidence, 0.9)
        self.assertGreater(result.normal[0], 0)
        self.assertAlmostEqual(np.linalg.norm(result.normal), 1)

    def test_ignores_a_stronger_horizontal_plane(self):
        result = prediction_from_planes([[0, 1, 0], [1, 0, 0]], [0.99, 0.4])
        self.assertFalse(result.symmetric_division)
        self.assertEqual(result.confidence, 0.4)
        self.assertEqual(result.normal, (1.0, 0.0, 0.0))

    def test_horizontal_only_is_not_a_left_right_half_mesh(self):
        self.assertEqual(prediction_from_planes([[0, 1, 0]], [0.99]), SymmetryPrediction(False, 0.0, None))

    def test_threshold_is_explicit_and_not_every_crop_is_approved(self):
        self.assertFalse(prediction_from_planes([[1, 0, 0]], [0.499]).symmetric_division)
        self.assertTrue(prediction_from_planes([[1, 0, 0]], [0.5]).symmetric_division)

    def test_malformed_model_outputs_never_approve(self):
        for normals, scores in (([], []), ([[1, 0]], [0.9]), ([[0, 0, 0]], [0.9]),
                                ([[1, 0, 0]], [float("nan")]), ([[1, 0, 0]], [1.1]),
                                ([[1, 0, 0]], [True]), ([[1, 0, 0]], ["0.9"]),
                                ([[1, 0, 0]], None), ([[float("inf"), 0, 0]], [0.8])):
            with self.subTest(normals=normals, scores=scores), self.assertRaises(Reflect3DError):
                prediction_from_planes(normals, scores)

    def test_desktop_backend_import_does_not_import_torch(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import housemaker.reflect3d_reference_symmetry; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


# ### Resident worker lifecycle without any native model calls ###
class Reflect3DSessionTests(unittest.TestCase):
    def setUp(self):
        self.lock = Lock()
        self.patchers = [
            patch("housemaker.reflect3d_reference_symmetry.ensure_reflect3d_model", return_value=Reflect3DBundle(
                Path("fake-reflect3d"), Path("fake-python"), Path("fake-dependencies"),
            )),
            patch("housemaker.qwen_plan_image_correction.QWEN_INFERENCE_LOCK", self.lock),
            patch("housemaker.reflect3d_reference_symmetry.stop_reflect3d_process", side_effect=self._stop),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _stop(process):
        process.returncode = 0

    def _worker(self, *messages, device="cuda"):
        process = FakeProcess([{"event": "ready", "device": device}, *messages])
        patcher = patch("housemaker.reflect3d_reference_symmetry.subprocess.Popen", return_value=process)
        factory = patcher.start()
        self.addCleanup(patcher.stop)
        return process, factory

    def test_one_model_serves_multiple_predictions_and_releases_gpu_on_exit(self):
        process, factory = self._worker(_result(1), _result(2))
        with Reflect3DSymmetrySession() as session:
            self.assertTrue(self.lock.locked())
            self.assertTrue(session.predict(CROP).symmetric_division)
            self.assertTrue(session.predict(CROP).symmetric_division)
            requests = [json.loads(line) for line in process.stdin.getvalue().splitlines()]
            self.assertEqual([item["id"] for item in requests], [1, 2])
            pixels = np.frombuffer(base64.b64decode(requests[0]["pixels"]), np.uint8).reshape(224, 224, 3)
            np.testing.assert_array_equal(pixels[112, 112], [30, 20, 10])
            self.assertEqual(session.stats.inference_count, 2)
            self.assertAlmostEqual(session.stats.model_inference_seconds, 0.05)
            self.assertGreater(session.stats.startup_seconds, 0)
            self.assertGreater(session.stats.inference_seconds, 0)
        self.assertFalse(self.lock.locked())
        self.assertIsNone(session._process)
        self.assertEqual(factory.call_count, 1)
        self.assertGreater(session.stats.close_seconds, 0)

    def test_cpu_startup_does_not_keep_the_shared_gpu_lock(self):
        self._worker(_result(), device="cpu")
        with Reflect3DSymmetrySession() as session:
            self.assertEqual(session.stats.device, "cpu")
            self.assertFalse(self.lock.locked())
            session.predict(CROP)

    def test_native_worker_is_offline_and_uses_only_its_own_dependencies(self):
        _, factory = self._worker()
        with Reflect3DSymmetrySession():
            env = factory.call_args.kwargs["env"]
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            self.assertIn("fake-dependencies", env["PYTHONPATH"])
            self.assertIn("housemaker.reflect3d_reference_worker", factory.call_args.args[0])

    def test_worker_start_failure_releases_gpu_lock(self):
        process = FakeProcess([{"event": "error", "error": "wrong checkpoint shape"}])
        with patch("housemaker.reflect3d_reference_symmetry.subprocess.Popen", return_value=process), self.assertRaisesRegex(
            Reflect3DError, "wrong checkpoint",
        ), Reflect3DSymmetrySession():
            self.fail("A broken detector cannot enter a session.")
        self.assertFalse(self.lock.locked())
        self.assertEqual(process.returncode, 0)

    def test_startup_timeout_stops_process_and_releases_gpu_lock(self):
        process, _ = self._worker()
        with patch("housemaker.reflect3d_reference_symmetry.REFLECT3D_STARTUP_TIMEOUT_SECONDS", 0), self.assertRaisesRegex(
            Reflect3DError, "timed out waiting for ready",
        ), Reflect3DSymmetrySession():
            self.fail("Timed-out startup must not enter a session.")
        self.assertFalse(self.lock.locked())
        self.assertEqual(process.returncode, 0)

    def test_cancelling_gpu_wait_does_not_release_another_jobs_lock(self):
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls >= 2

        self.lock.acquire()
        try:
            with self.assertRaises(Reflect3DCancelled), Reflect3DSymmetrySession(cancellation_check=cancelled):
                self.fail("Cancelled startup must not enter a session.")
            self.assertTrue(self.lock.locked())
        finally:
            self.lock.release()

    def test_process_crash_preserves_stderr_reason_and_releases_gpu_lock(self):
        process = FakeProcess([])
        process.stderr = io.StringIO("RuntimeError: checkpoint has an invalid shape\n")
        process.returncode = 1
        with patch("housemaker.reflect3d_reference_symmetry.subprocess.Popen", return_value=process), self.assertRaisesRegex(
            Reflect3DError, "checkpoint has an invalid shape",
        ), Reflect3DSymmetrySession():
            self.fail("Crashed startup must not enter a session.")
        self.assertFalse(self.lock.locked())

    def test_cancelling_a_prediction_closes_the_resident_worker(self):
        cancelled = False
        self._worker(_result())
        with Reflect3DSymmetrySession(cancellation_check=lambda: cancelled) as session:
            cancelled = True
            with self.assertRaises(Reflect3DCancelled):
                session.predict(CROP)
            self.assertIsNone(session._process)
            self.assertFalse(self.lock.locked())

    def test_protocol_failure_and_stale_or_boolean_ids_close_the_worker(self):
        for message in (_result(99), _result(True), _result(normals=[[0, 0, 0]]),
                        {"event": "error", "error": "out of memory"}):
            with self.subTest(message=message):
                self._worker(message)
                with Reflect3DSymmetrySession() as session:
                    with self.assertRaises(Reflect3DError):
                        session.predict(CROP)
                    self.assertFalse(self.lock.locked())
                    self.assertIsNone(session._process)

    def test_predict_requires_a_running_session(self):
        with self.assertRaisesRegex(Reflect3DError, "not running"):
            Reflect3DSymmetrySession().predict(CROP)

    def test_public_helper_returns_generated_reference_contract_and_releases_process(self):
        process, factory = self._worker(_result())
        result = infer_reflect3d_symmetry(CROP)
        self.assertEqual(result["symmetric"], True)
        self.assertEqual(result["confidence"], 0.8)
        self.assertIn("minor asymmetry", result["reason"])
        self.assertFalse(self.lock.locked())
        self.assertEqual(process.returncode, 0)
        self.assertEqual(factory.call_count, 1)

    def test_public_helper_reports_unsuitable_plane_and_low_confidence(self):
        for normals, probabilities, fragment in (
            ([[0, 1, 0]], [0.9], "did not detect"),
            ([[1, 0, 0]], [0.3], "below 0.50"),
        ):
            with self.subTest(normals=normals):
                self._worker(_result(normals=normals, probabilities=probabilities))
                result = infer_reflect3d_symmetry(CROP)
                self.assertFalse(result["symmetric"])
                self.assertIn(fragment, result["reason"])
                self.assertFalse(self.lock.locked())

    def test_invalid_input_is_rejected_before_loading_optional_model(self):
        with patch("housemaker.reflect3d_reference_symmetry.ensure_reflect3d_model") as setup, self.assertRaises(Reflect3DError):
            infer_reflect3d_symmetry(np.zeros((1, 1, 4), np.uint8))
        setup.assert_not_called()

    def test_already_cancelled_does_not_start_model(self):
        with patch("housemaker.reflect3d_reference_symmetry.ensure_reflect3d_model") as setup, self.assertRaises(Reflect3DCancelled):
            infer_reflect3d_symmetry(CROP, cancellation_check=lambda: True)
        setup.assert_not_called()


# ### Pinned downloads and dependency isolation ###
class Reflect3DSetupTests(unittest.TestCase):
    def test_checkpoint_and_sources_have_real_pinned_hashes(self):
        self.assertEqual(len(REFLECT3D_WEIGHTS_SHA256), 64)
        self.assertEqual(len(REFLECT3D_SOURCE_SHA256), 5)
        self.assertTrue(all(len(value) == 64 for value in REFLECT3D_SOURCE_SHA256.values()))

    def test_corrupt_existing_asset_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(Reflect3DError, "checksum mismatch"):
                _verified_asset(path, hashlib.sha256(b"good").hexdigest())
            self.assertEqual(path.read_bytes(), b"bad")

    def test_failed_download_checksum_removes_only_its_partial_file(self):
        response = io.BytesIO(b"bad")
        response.headers = {"Content-Type": "application/octet-stream"}
        with tempfile.TemporaryDirectory() as temporary, patch(
            "housemaker.reflect3d_reference_setup.urlopen", return_value=response,
        ):
            path = Path(temporary) / "model"
            with self.assertRaisesRegex(Reflect3DError, "SHA-256"):
                _download_asset("https://example.invalid/model", path, hashlib.sha256(b"good").hexdigest())
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".download").exists())

    def test_setup_reuses_cuda_python_read_only_and_targets_own_dependency_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "reflect3d"
            with patch("housemaker.reflect3d_reference_setup._has_compatible_torch", side_effect=[False, True]), patch(
                "housemaker.reflect3d_reference_setup._run_setup",
            ) as run:
                python, dependencies = _prepare_runtime(directory)
            self.assertIn("cubicasa5k", str(python))
            self.assertEqual(dependencies, directory / "dependencies")
            command = run.call_args.args[0]
            self.assertIn("--target", command)
            self.assertEqual(command[command.index("--target") + 1], str(dependencies))
            self.assertNotIn("torch==2.6.0", command)

    def test_lazy_setup_records_unspecified_license_not_invented_redistribution_rights(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "housemaker.reflect3d_reference_setup.default_reflect3d_directory", return_value=Path(temporary),
        ), patch("housemaker.reflect3d_reference_setup._download_asset") as download, patch(
            "housemaker.reflect3d_reference_setup._prepare_runtime", return_value=(Path("python"), Path("dependencies")),
        ):
            bundle = ensure_reflect3d_model()
            self.assertEqual(download.call_count, 6)
            manifest = json.loads((bundle.directory / "reflect3d-manifest.json").read_text())
            self.assertEqual(manifest["license"], "unspecified by upstream")
            self.assertEqual(manifest["acceptance_policy"]["confidence_threshold"], 0.5)
            self.assertEqual(manifest["acceptance_policy"]["max_vertical_normal_y"], 0.4)
            self.assertIn("Obtain the authors' permission", (bundle.directory / "NOTICE.txt").read_text())


# ### Direct CPU-only test entry point ###
if __name__ == "__main__":
    unittest.main()
