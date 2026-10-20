"""Compare the existing local symmetry backends, without downloads or API calls."""

# ### Imports ###
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from housemaker.reference_symmetry_inference import (
    SYMMETRY_MODEL_QWEN,
    SYMMETRY_MODEL_REFLECT3D,
    open_reference_symmetry_session,
)

# ### Benchmark policy ###
LOCAL_MODELS = (SYMMETRY_MODEL_QWEN, SYMMETRY_MODEL_REFLECT3D)
DEFAULT_CALL_TIMEOUT_SECONDS = 180.0
DEFAULT_SESSION_TIMEOUT_SECONDS = 300.0
GPU_SAMPLE_INTERVAL_SECONDS = 1.0


# ### Input validation and reproducibility ###
def load_inputs(
    paths: list[Path], *, prepare_reference: bool = False
) -> list[tuple[dict, np.ndarray]]:
    """Decode once, before either timed model run, with no image alteration."""
    if len(paths) < 2:
        raise ValueError(
            "Provide at least two images to measure cold and warm requests."
        )
    inputs = []
    for path in paths:
        path = path.resolve(strict=True)
        raw = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"A BGR or BGRA reference image is required: {path}")
        source_shape = image.shape
        if prepare_reference:
            from housemaker.reference_generation import (
                prepare_reference_generation_input,
            )

            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            payload = prepare_reference_generation_input(image)
            image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        metadata = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": image.shape[2],
            "source_shape": source_shape,
            "prepared_reference": prepare_reference,
            "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        }
        inputs.append((metadata, np.ascontiguousarray(image)))
    return inputs


def forbid_download(*_args, **_kwargs):
    raise RuntimeError("The benchmark requires already-installed local model assets.")


# ### Lightweight GPU observation, without importing Torch ###
def gpu_snapshot() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode:
            return []
        rows = []
        for line in result.stdout.splitlines():
            name, used, total, utilization = (part.strip() for part in line.split(","))
            rows.append(
                {
                    "name": name,
                    "memory_used_mib": int(used),
                    "memory_total_mib": int(total),
                    "utilization_percent": int(utilization),
                }
            )
        return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


class GpuSampler:
    def __init__(self):
        self.samples: list[list[dict]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            self.samples.append(gpu_snapshot())
            self.stop.wait(GPU_SAMPLE_INTERVAL_SECONDS)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=6)


# ### Session diagnostics and timed inference ###
def backend_diagnostics(session) -> dict:
    """Read diagnostics only; never change model settings or device selection."""
    backend = getattr(session, "_backend", None)
    if backend is None:
        return {}
    stats = getattr(backend, "stats", None)
    if stats is not None and is_dataclass(stats):
        return {"stats": asdict(stats), "device": stats.device}
    bundle = getattr(backend, "_bundle", None)
    result = {"backend": type(backend).__name__}
    if bundle is not None:
        result["language_model"] = bundle.language_model.name
        result["vision_projector"] = bundle.vision_projector.name
        # llama.cpp's CUDA status is observed from its own log, not assumed.
        temporary_directory = getattr(backend, "_temporary_directory", None)
        if temporary_directory is not None:
            log = Path(temporary_directory.name, "llama-server.log").read_text(
                encoding="utf-8",
                errors="replace",
            )
            result["cuda_reported_in_log"] = "CUDA" in log
    return result


def benchmark_model(
    model: str,
    inputs,
    *,
    call_timeout: float,
    session_timeout: float,
    session_factory=open_reference_symmetry_session,
) -> dict:
    if model not in LOCAL_MODELS:
        raise ValueError("Only local Qwen and Reflect3D may be benchmarked.")
    started = time.perf_counter()
    session_deadline = time.monotonic() + session_timeout
    call_deadline = session_deadline
    cancelled = lambda: time.monotonic() >= min(call_deadline, session_deadline)
    report = {"model": model, "requests": [], "gpu_before": gpu_snapshot()}
    session = session_factory(model, cancellation_check=cancelled)
    try:
        with GpuSampler() as sampler:
            session.__enter__()
            for metadata, image in inputs:
                call_deadline = min(session_deadline, time.monotonic() + call_timeout)
                call_started = time.perf_counter()
                request = {
                    "input_sha256": metadata["sha256"],
                    "input_path": metadata["path"],
                }
                try:
                    result = session.infer(image, cancellation_check=cancelled)
                    request.update(asdict(result))
                except Exception as error:  # noqa: BLE001 - report each explicit model failure.
                    request["error"] = f"{type(error).__name__}: {error}"
                request["wall_seconds"] = time.perf_counter() - call_started
                report["requests"].append(request)
                print(
                    json.dumps({"event": "request", "model": model, **request}),
                    flush=True,
                )
                if "error" in request or cancelled():
                    break
            report["backend"] = backend_diagnostics(session)
        report["gpu_peak_used_mib"] = max(
            (gpu["memory_used_mib"] for sample in sampler.samples for gpu in sample),
            default=None,
        )
    finally:
        close_started = time.perf_counter()
        session.close()
        report["close_seconds"] = time.perf_counter() - close_started
        report["total_seconds"] = time.perf_counter() - started
        report["gpu_after"] = gpu_snapshot()
    successful = [item for item in report["requests"] if "error" not in item]
    report["completed_requests"] = len(successful)
    report["cold_first_request_seconds"] = (
        report["requests"][0]["wall_seconds"] if report["requests"] else None
    )
    warm = [
        item["wall_seconds"] for item in report["requests"][1:] if "error" not in item
    ]
    report["warm_median_seconds"] = statistics.median(warm) if warm else None
    return report


# ### Command-line entry point ###
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prepare-reference",
        action="store_true",
        help="Use the app's 1024px white reference canvas before timing.",
    )
    parser.add_argument("--rounds", type=int, choices=range(1, 4), default=1)
    parser.add_argument(
        "--models", nargs="+", choices=LOCAL_MODELS, default=list(LOCAL_MODELS)
    )
    parser.add_argument(
        "--call-timeout", type=float, default=DEFAULT_CALL_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--session-timeout", type=float, default=DEFAULT_SESSION_TIMEOUT_SECONDS
    )
    args = parser.parse_args()
    if not (0 < args.call_timeout <= 180 and 0 < args.session_timeout <= 300):
        parser.error("Call timeout must be <= 180 s and session timeout <= 300 s.")
    inputs = load_inputs(args.images, prepare_reference=args.prepare_reference)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_threads": os.cpu_count(),
        "inputs": [metadata for metadata, _image in inputs],
        "rounds": args.rounds,
        "policy": "Unchanged app session API; sequential models; first request includes lazy loading; no downloads.",
        "models": [],
    }
    with ExitStack() as stack:
        stack.enter_context(
            patch("housemaker.reflect3d_reference_setup.urlopen", forbid_download)
        )
        stack.enter_context(
            patch("housemaker.reflect3d_reference_setup._run_setup", forbid_download)
        )
        for model in args.models:
            report["models"].append(
                benchmark_model(
                    model,
                    inputs * args.rounds,
                    call_timeout=args.call_timeout,
                    session_timeout=args.session_timeout,
                )
            )
            args.output.write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
            )
    print(f"Report: {args.output.resolve()}", flush=True)
    return int(
        any(
            item["completed_requests"] != len(inputs) * args.rounds
            for item in report["models"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
