"""Cancellable local symmetry detection without importing Torch in the UI."""

# ### Imports ###
from __future__ import annotations

import base64
import json
import math
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .reflect3d_reference_setup import (
    REFLECT3D_MODEL_ID,
    Reflect3DCancelled,
    Reflect3DError,
    check_reflect3d_cancelled,
    ensure_reflect3d_model,
    stop_reflect3d_process,
)

__all__ = [
    "REFLECT3D_MODEL_ID", "Reflect3DCancelled", "Reflect3DError",
    "Reflect3DSymmetryCancelled", "Reflect3DSymmetryError", "Reflect3DSymmetrySession",
    "Reflect3DSymmetryStats", "SymmetryPrediction", "infer_reflect3d_symmetry",
    "prediction_from_planes", "prepare_reflect3d_pixels", "symmetry_payload_from_prediction",
]

# ### Explicit initial acceptance policy ###
REFLECT3D_INPUT_SIZE = 224
REFLECT3D_CONFIDENCE_THRESHOLD = 0.5
REFLECT3D_MAX_VERTICAL_NORMAL_Y = 0.4
REFLECT3D_POLICY_VERSION = 1
REFLECT3D_STARTUP_TIMEOUT_SECONDS = 180.0
REFLECT3D_REQUEST_TIMEOUT_SECONDS = 120.0
REFLECT3D_PREPROCESSING_VERSION = "rgba-bbox-square-white-1.2-bicubic-imagenet-v1"


# ### Public predictions and actual timing counters ###
@dataclass(frozen=True)
class SymmetryPrediction:
    symmetric_division: bool
    confidence: float
    normal: tuple[float, float, float] | None


@dataclass
class Reflect3DSymmetryStats:
    startup_seconds: float = 0.0
    inference_seconds: float = 0.0
    model_inference_seconds: float = 0.0
    inference_count: int = 0
    close_seconds: float = 0.0
    device: str = ""


# ### Aspect-preserving input and camera-space plane selection ###
def reflect3d_policy_signature() -> dict:
    return {"version": REFLECT3D_POLICY_VERSION,
            "confidence_threshold": REFLECT3D_CONFIDENCE_THRESHOLD,
            "max_vertical_normal_y": REFLECT3D_MAX_VERTICAL_NORMAL_Y,
            "preprocessing": REFLECT3D_PREPROCESSING_VERSION}


def prepare_reflect3d_pixels(crop_bgr: np.ndarray) -> np.ndarray:
    """Use a known alpha silhouette if supplied; never invent segmentation.

    The original RGB demo accidentally treats the blue channel as alpha. This
    follows its intended 1.2x square-crop/ImageNet preprocessing while preserving
    actual opaque pixels and genuine alpha instead. BGR crops use their full box.
    """
    crop = np.asarray(crop_bgr)
    if (crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] not in (3, 4)
            or not crop.shape[0] or not crop.shape[1] or max(crop.shape[:2]) > 16_384):
        raise Reflect3DError("Reflect3D needs a nonempty uint8 BGR or BGRA object crop.")
    rgb = np.ascontiguousarray(crop[:, :, :3][:, :, ::-1])
    height, width = crop.shape[:2]
    left, top, right, bottom = 0, 0, width, height
    if crop.shape[2] == 4:
        alpha = crop[:, :, 3].astype(np.float32) / 255.0
        ys, xs = np.nonzero(alpha > 0.5)
        if not xs.size:
            raise Reflect3DError("Reflect3D cannot analyse a fully transparent object crop.")
        left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        rgb = np.rint(rgb * alpha[:, :, None] + 255 * (1 - alpha[:, :, None])).astype(np.uint8)
    extent = max(1, math.ceil(max(right - left, bottom - top) * 1.2))
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    x, y = math.floor(center_x - extent / 2), math.floor(center_y - extent / 2)
    square = Image.new("RGB", (extent, extent), "white")
    square.paste(Image.fromarray(rgb), (-x, -y))
    square = square.resize((REFLECT3D_INPUT_SIZE, REFLECT3D_INPUT_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(square, dtype=np.uint8).copy()


def prediction_from_planes(normals, probabilities) -> SymmetryPrediction:
    """Accept the strongest upright reflection plane, including soft symmetry.

    Scores are the author's sigmoid outputs, not calibrated probabilities.
    A horizontal plane cannot justify left/right half-mesh mirroring. Camera Y
    is image vertical; the tolerance admits modest camera tilt. A plane's sign
    does not change its reflection, so returned normals use a stable sign.
    """
    try:
        normals, scores = np.asarray(normals), np.asarray(probabilities)
        if normals.dtype.kind not in "iuf" or scores.dtype.kind not in "iuf":
            raise ValueError("reflection outputs must be numeric, not booleans or strings")
        normals = normals.astype(np.float64)
        scores = scores.astype(np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise Reflect3DError("Reflect3D returned nonnumeric reflection-plane predictions.") from exc
    if (normals.ndim != 2 or normals.shape[1] != 3 or normals.shape[0] != scores.size
            or not scores.size or not np.isfinite(normals).all() or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))):
        raise Reflect3DError("Reflect3D returned malformed reflection-plane predictions.")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-8):
        raise Reflect3DError("Reflect3D returned an invalid zero-length reflection normal.")
    normals = normals / lengths[:, None]
    upright = np.flatnonzero(np.abs(normals[:, 1]) <= REFLECT3D_MAX_VERTICAL_NORMAL_Y)
    if not upright.size:
        return SymmetryPrediction(False, 0.0, None)
    index = int(upright[np.argmax(scores[upright])])
    normal = normals[index]
    if normal[0] < -1e-8 or (abs(normal[0]) <= 1e-8 and normal[2] < 0):
        normal = -normal
    confidence = float(scores[index])
    return SymmetryPrediction(
        confidence >= REFLECT3D_CONFIDENCE_THRESHOLD, confidence, tuple(float(value) for value in normal),
    )


# ### Resident isolated worker and cooperative GPU ownership ###
class Reflect3DSymmetrySession:
    """Keep one feed-forward model resident; close before another GPU phase."""

    def __init__(self, *, cancellation_check=None, progress_callback=None):
        self.cancellation_check, self.progress_callback = cancellation_check, progress_callback
        self.stats = Reflect3DSymmetryStats()
        self._process = None
        self._messages: queue.Queue[dict] = queue.Queue()
        self._readers: list[threading.Thread] = []
        self._stderr_tail: list[str] = []
        self._gpu_lock = None
        self._request_id = 0

    def __enter__(self):
        try:
            return self._open()
        except BaseException:
            self.close()
            raise

    def _open(self):
        if self._process is not None:
            return self
        start = time.perf_counter()
        check_reflect3d_cancelled(self.cancellation_check)
        bundle = ensure_reflect3d_model(
            cancellation_check=self.cancellation_check, progress_callback=self.progress_callback,
        )
        from .qwen_plan_image_correction import QWEN_INFERENCE_LOCK

        while not QWEN_INFERENCE_LOCK.acquire(timeout=0.1):
            check_reflect3d_cancelled(self.cancellation_check)
        self._gpu_lock = QWEN_INFERENCE_LOCK
        check_reflect3d_cancelled(self.cancellation_check)
        environment = os.environ.copy()
        environment.update({
            "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": os.pathsep.join((str(bundle.dependencies), str(Path(__file__).resolve().parents[1]))),
        })
        self._messages = queue.Queue()
        self._stderr_tail = []
        self._process = subprocess.Popen(
            [str(bundle.python), "-m", "housemaker.reflect3d_reference_worker", "--bundle", str(bundle.directory)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._readers = [threading.Thread(target=self._read_stdout, daemon=True),
                         threading.Thread(target=self._read_stderr, daemon=True)]
        for reader in self._readers:
            reader.start()
        ready = self._receive("ready", REFLECT3D_STARTUP_TIMEOUT_SECONDS)
        if ready.get("device") not in {"cpu", "cuda"}:
            raise Reflect3DError("Reflect3D returned an invalid startup response.")
        self.stats.device = ready["device"]
        if self.stats.device == "cpu":
            self._release_gpu_lock()
        self.stats.startup_seconds += time.perf_counter() - start
        if self.progress_callback:
            self.progress_callback(f"Reflect3D local symmetry ready on {self.stats.device}.")
        return self

    def _read_stdout(self) -> None:
        process = self._process
        if process is not None and process.stdout is not None:
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    self._stderr_tail = [*self._stderr_tail[-19:], line.strip()[:500]]
                    continue
                if isinstance(message, dict):
                    self._messages.put(message)

    def _read_stderr(self) -> None:
        process = self._process
        if process is not None and process.stderr is not None:
            for line in process.stderr:
                self._stderr_tail = [*self._stderr_tail[-19:], line.strip()[:500]]

    def _receive(self, expected: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            check_reflect3d_cancelled(self.cancellation_check)
            try:
                message = self._messages.get(timeout=0.1)
            except queue.Empty:
                if self._process is None or self._process.poll() is not None:
                    raise Reflect3DError("Reflect3D worker stopped: " + " ".join(self._stderr_tail[-4:]))
                continue
            if message.get("event") == "error":
                raise Reflect3DError(f"Local Reflect3D inference failed: {message.get('error', 'unknown error')}")
            if message.get("event") != expected:
                raise Reflect3DError("Reflect3D returned an unexpected worker response.")
            return message
        raise Reflect3DError(f"Reflect3D timed out waiting for {expected}.")

    def predict(self, crop_bgr: np.ndarray) -> SymmetryPrediction:
        try:
            check_reflect3d_cancelled(self.cancellation_check)
        except Reflect3DCancelled:
            self.close()
            raise
        pixels = prepare_reflect3d_pixels(crop_bgr)
        if self._process is None or self._process.stdin is None:
            raise Reflect3DError("The Reflect3D symmetry session is not running.")
        self._request_id += 1
        request = {"id": self._request_id, "pixels": base64.b64encode(pixels.tobytes()).decode("ascii")}
        start = time.perf_counter()
        try:
            self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            response = self._receive("result", REFLECT3D_REQUEST_TIMEOUT_SECONDS)
            if type(response.get("id")) is not int or response["id"] != self._request_id:
                raise Reflect3DError("Reflect3D returned a stale prediction response.")
            prediction = prediction_from_planes(response.get("normals"), response.get("probabilities"))
            model_seconds = response.get("inference_seconds")
            if isinstance(model_seconds, bool) or not isinstance(model_seconds, (int, float)) or not math.isfinite(model_seconds) or model_seconds < 0:
                raise Reflect3DError("Reflect3D returned an invalid inference timing.")
            check_reflect3d_cancelled(self.cancellation_check)
            self.stats.inference_count += 1
            self.stats.inference_seconds += time.perf_counter() - start
            self.stats.model_inference_seconds += model_seconds
            return prediction
        except (OSError, Reflect3DError):
            # Cancellation or a broken protocol must not leave a GPU owner alive.
            self.close()
            raise

    def _release_gpu_lock(self) -> None:
        lock, self._gpu_lock = self._gpu_lock, None
        if lock is not None:
            lock.release()

    def close(self) -> None:
        start = time.perf_counter()
        process, self._process = self._process, None
        try:
            if process is not None:
                stop_reflect3d_process(process)
                for reader in self._readers:
                    reader.join(timeout=2)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        finally:
            self._readers = []
            self._release_gpu_lock()
            if process is not None:
                self.stats.close_seconds += time.perf_counter() - start

    def __exit__(self, *_args):
        self.close()


# ### Generated-reference symmetry entry point ###
Reflect3DSymmetryError = Reflect3DError
Reflect3DSymmetryCancelled = Reflect3DCancelled


def infer_reflect3d_symmetry(
    image_bgra: np.ndarray, *, cancellation_check=None, progress_callback=None,
) -> dict:
    """Infer approximate upright reflection symmetry from the generated image.

    Only Reflect3D's learned reflection planes decide this result; no pixel
    similarity heuristic or language-model fallback is used. The score is the
    model's sigmoid output, not a calibrated probability.
    """
    check_reflect3d_cancelled(cancellation_check)
    # Validate before downloading optional weights or starting a heavy runtime.
    prepare_reflect3d_pixels(image_bgra)
    try:
        with Reflect3DSymmetrySession(
            cancellation_check=cancellation_check, progress_callback=progress_callback,
        ) as session:
            prediction = session.predict(image_bgra)
        check_reflect3d_cancelled(cancellation_check)
    except Reflect3DError:
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise Reflect3DError(f"Local Reflect3D symmetry inference failed: {exc}") from exc
    return symmetry_payload_from_prediction(prediction)


def symmetry_payload_from_prediction(prediction: SymmetryPrediction) -> dict:
    """Describe the same learned-plane decision for one-shot and resident use."""
    if prediction.symmetric_division:
        reason = (
            "Reflect3D detected an upright reflection plane compatible with "
            "approximate bilateral symmetry; minor asymmetry is tolerated."
        )
    elif prediction.normal is None:
        reason = "Reflect3D did not detect an upright reflection plane suitable for symmetric division."
    else:
        reason = (
            f"Reflect3D's strongest upright reflection score "
            f"({prediction.confidence:.3f}) was below {REFLECT3D_CONFIDENCE_THRESHOLD:.2f}."
        )
    return {
        "symmetric": prediction.symmetric_division,
        "confidence": prediction.confidence,
        "reason": reason,
    }
