# ### Imports ###
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from http.client import HTTPConnection, HTTPException
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from . import qwen_plan_image_correction as qwen
from . import stable_diffusion_reference as sdxl
from .reference_generation_models import DEFAULT_REFERENCE_NEGATIVE_PROMPT

# ### Constants and types ###
# The bundled stable-diffusion.cpp c92d73c server owns one sd_ctx for its lifetime.
# Native requests keep prompts literal and distinguish Qwen references from SDXL
# init images. No denoising-step cache or approximate quality setting is enabled.
SERVER_FILENAME = "sd-server.exe" if os.name == "nt" else "sd-server"
SERVER_STARTUP_TIMEOUT_SECONDS = 600.0
GENERATION_TIMEOUT_SECONDS = 7200.0
HTTP_TIMEOUT_SECONDS = 5.0
POLL_SECONDS = 0.10
MAX_RESPONSE_BYTES = 96 * 1024 * 1024
CancellationCheck = Callable[[], bool]
ProgressCallback = Callable[[str, int], None]


class LocalReferenceSessionError(RuntimeError):
    """A resident-process failure, without private prompt or filesystem details."""


class LocalReferenceSessionCancelled(LocalReferenceSessionError):
    pass


# ### Shared resident server ###
class _ResidentSdCppServer:
    def __init__(
        self,
        command_factory: Callable[[], list[str]],
        *,
        label: str,
        cancellation_check: CancellationCheck | None = None,
    ) -> None:
        self._command_factory = command_factory
        self._label = label
        self._cancellation_check = cancellation_check
        self._process: subprocess.Popen[bytes] | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._log_file: BinaryIO | None = None
        self._log_path: Path | None = None
        self._port = 0
        self._gpu_lock: threading.Lock | None = None
        self._request_lock = threading.Lock()
        self._wake = threading.Event()

    def __enter__(self) -> _ResidentSdCppServer:  # noqa: PYI034 - Python 3.10
        # Defer downloading and loading until the first actual request.
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def generate(
        self,
        payload: dict[str, object],
        *,
        cancellation_check: CancellationCheck | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> bytes:
        check = cancellation_check if cancellation_check is not None else self._cancellation_check
        self._acquire(self._request_lock, check)
        try:
            self._check_cancelled(check)
            self._ensure_started(check, progress_callback)
            _report(progress_callback, f"Generating with resident {self._label}", 0)
            submitted = self._request("POST", "/sdcpp/v1/img_gen", payload)
            job_id = submitted.get("id")
            if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", job_id):
                raise LocalReferenceSessionError("The local image server returned an invalid job identifier.")
            deadline = time.monotonic() + GENERATION_TIMEOUT_SECONDS
            while True:
                self._check_cancelled(check)
                self._check_process()
                if time.monotonic() > deadline:
                    raise LocalReferenceSessionError("The local image-generation job timed out.")
                job = self._request("GET", f"/sdcpp/v1/jobs/{job_id}")
                status = job.get("status")
                if status == "completed":
                    result = _decode_job_output(job)
                    self._check_cancelled(check)
                    _report(progress_callback, "Reference generation complete", 100)
                    return result
                if status in {"failed", "cancelled"}:
                    raise LocalReferenceSessionError(self._failure_reason("generation failed"))
                if status not in {"queued", "generating"}:
                    raise LocalReferenceSessionError("The local image server returned an invalid job status.")
                self._wake.wait(POLL_SECONDS)
        except BaseException:
            # The native runtime cannot interrupt an active generation. Stop its
            # process, not merely HTTP polling. A later item may lazily reload it.
            self.close()
            raise
        finally:
            self._request_lock.release()

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            qwen._terminate_process(process)
            if process.poll() is None:
                # Never let another large model start while an unkillable
                # native process may still hold GPU memory. Keep state so an
                # explicit later close can retry stopping it.
                raise LocalReferenceSessionError("The resident image process could not be stopped safely.")
        self._process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
        self._log_path = None
        if self._gpu_lock is not None:
            self._gpu_lock.release()
            self._gpu_lock = None

    def _ensure_started(
        self,
        check: CancellationCheck | None,
        progress_callback: ProgressCallback | None,
    ) -> None:
        if self._process is not None:
            self._check_process()
            return
        command = self._command_factory()
        executable = Path(command[0])
        if not executable.is_file():
            raise LocalReferenceSessionError(
                "The resident local image runtime (sd-server) is missing. "
                "Reinstall the bundled Qwen/Stable Diffusion runtime."
            )
        _report(progress_callback, "Waiting for the local image-generation GPU", 0)
        self._acquire(qwen.QWEN_INFERENCE_LOCK, check)
        self._gpu_lock = qwen.QWEN_INFERENCE_LOCK
        self._check_cancelled(check)
        self._port = _unused_loopback_port()
        self._directory = tempfile.TemporaryDirectory(prefix="housemaker-reference-server-")
        self._log_path = Path(self._directory.name) / "sd-server.log"
        self._log_file = self._log_path.open("wb")
        _report(progress_callback, f"Loading {self._label} once for this reference group", 0)
        try:
            self._process = subprocess.Popen(
                [*command, "--listen-ip", "127.0.0.1", "--listen-port", str(self._port)],
                cwd=str(executable.parent),
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=qwen._hidden_process_flags(),
            )
        except (OSError, subprocess.SubprocessError):
            raise LocalReferenceSessionError("The resident local image process could not be started.") from None
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_SECONDS
        while True:
            self._check_cancelled(check)
            self._check_process()
            if time.monotonic() > deadline:
                raise LocalReferenceSessionError("Loading the resident local image model timed out.")
            try:
                capabilities = self._request("GET", "/sdcpp/v1/capabilities")
            except (ConnectionError, TimeoutError, OSError, HTTPException):
                self._wake.wait(POLL_SECONDS)
                continue
            if capabilities.get("current_mode") != "img_gen":
                raise LocalReferenceSessionError("The local image runtime does not support native image requests.")
            return

    def _request(
        self, method: str, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        # Direct loopback HTTP deliberately ignores proxy environment variables.
        connection = HTTPConnection("127.0.0.1", self._port, timeout=HTTP_TIMEOUT_SECONDS)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        try:
            connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            content = response.read(MAX_RESPONSE_BYTES + 1)
            if len(content) > MAX_RESPONSE_BYTES:
                raise LocalReferenceSessionError("The local image server response was too large.")
            if response.status not in {200, 202}:
                raise LocalReferenceSessionError(
                    f"The resident local image request failed (HTTP {response.status})."
                )
            decoded = json.loads(content)
            if not isinstance(decoded, dict):
                raise TypeError("Expected an object")
            return decoded
        except (UnicodeError, ValueError, TypeError):
            raise LocalReferenceSessionError("The local image server returned invalid JSON.") from None
        finally:
            connection.close()

    def _check_process(self) -> None:
        if self._process is None or self._process.poll() is not None:
            raise LocalReferenceSessionError(self._failure_reason("process stopped unexpectedly"))

    def _failure_reason(self, default: str) -> str:
        try:
            if self._log_path is None:
                raise OSError("No log")
            with self._log_path.open("rb") as handle:
                handle.seek(max(0, self._log_path.stat().st_size - 16_384))
                tail = handle.read().decode("utf-8", errors="replace").lower()
        except OSError:
            tail = ""
        reasons = (
            (("out of memory", "failed to allocate", "cuda_error_out_of_memory"), "insufficient GPU or system memory"),
            (("unknown argument", "unknown option", "unrecognized option"), "the installed runtime does not support resident generation"),
            (("failed to load", "new_sd_ctx", "unsupported model", "invalid model"), "the selected model could not be loaded"),
            (("cuda error", "cudaerror", "cuda driver", "no cuda-capable"), "the CUDA runtime or GPU is unavailable"),
        )
        reason = next((reason for needles, reason in reasons if any(needle in tail for needle in needles)), default)
        return f"Resident {self._label} reference generation failed: {reason}."

    @staticmethod
    def _check_cancelled(check: CancellationCheck | None) -> None:
        if check is not None and check():
            raise LocalReferenceSessionCancelled("Local reference generation was cancelled.")

    @classmethod
    def _acquire(cls, lock: threading.Lock, check: CancellationCheck | None) -> None:
        while not lock.acquire(timeout=POLL_SECONDS):
            cls._check_cancelled(check)
        try:
            cls._check_cancelled(check)
        except BaseException:
            lock.release()
            raise


# ### Qwen reference session ###
class QwenReferenceImageSession:
    """Reuse the exact local Qwen image bundle throughout one image phase."""

    def __init__(
        self,
        *,
        cancellation_check: CancellationCheck | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._progress_callback = progress_callback
        self._server = _ResidentSdCppServer(
            self._command, label="Qwen-Image-2.1", cancellation_check=cancellation_check
        )

    def __enter__(self) -> QwenReferenceImageSession:  # noqa: PYI034 - Python 3.10
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._server.close()

    def __call__(
        self,
        image_bytes: bytes,
        *,
        api_key: str = "",
        prompt: str,
        output_size: tuple[int, int],
        cancellation_check: CancellationCheck | None = None,
    ) -> bytes:
        del api_key
        prompt = qwen._validate_prompt(prompt)
        output_size = qwen._validate_output_size(output_size)
        prepared = qwen._decode_prepared_png(image_bytes, output_size)
        payload = _image_request(prepared, prompt)
        payload["ref_images"] = [_encode_png(prepared)]
        try:
            result = self._server.generate(
                payload, cancellation_check=cancellation_check, progress_callback=self._progress_callback
            )
            return _validate_output_png(result, output_size)
        except LocalReferenceSessionCancelled as error:
            raise qwen.QwenPlanCorrectionCancelled(str(error)) from None
        except (LocalReferenceSessionError, OSError, HTTPException) as error:
            message = str(error) if isinstance(error, LocalReferenceSessionError) else "The resident Qwen image server connection failed."
            raise qwen.QwenPlanCorrectionError(message) from None

    @staticmethod
    def _command() -> list[str]:
        bundle = qwen._validate_bundle(qwen.default_qwen_model_bundle())
        return [
            str(bundle.executable.with_name(SERVER_FILENAME)),
            "--diffusion-model", str(bundle.diffusion_model),
            "--vae", str(bundle.vae_model), "--llm", str(bundle.llm_model),
            "--llm_vision", str(bundle.vision_model),
            "--cfg-scale", qwen.QWEN_CFG_SCALE,
            "--sampling-method", qwen.QWEN_SAMPLING_METHOD,
            "--steps", str(qwen.QWEN_DEFAULT_STEPS),
            "--offload-to-cpu", "--diffusion-fa", "--vae-tiling",
        ]


# ### Stable Diffusion reference session ###
class StableDiffusionReferenceSession:
    """Reuse the pinned SDXL checkpoint; preserve all existing img2img settings."""

    def __init__(
        self,
        *,
        cancellation_check: CancellationCheck | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._cancellation_check = cancellation_check
        self._active_check = cancellation_check
        self._progress_callback = progress_callback
        self._active_progress = progress_callback
        self._server = _ResidentSdCppServer(
            self._command, label="Stable Diffusion XL", cancellation_check=cancellation_check
        )

    def __enter__(self) -> StableDiffusionReferenceSession:  # noqa: PYI034 - Python 3.10
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._server.close()

    def __call__(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        negative_prompt: str = DEFAULT_REFERENCE_NEGATIVE_PROMPT,
        output_size: tuple[int, int] = (1024, 1024),
        strength: float = 0.45,
        cancellation_check: CancellationCheck | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> bytes:
        prepared, prompt, strength = sdxl._validate_inputs(image_bytes, prompt, output_size, strength)
        sdxl._validate_negative_prompt(negative_prompt)
        self._active_check = cancellation_check if cancellation_check is not None else self._cancellation_check
        self._active_progress = progress_callback if progress_callback is not None else self._progress_callback
        payload = _image_request(prepared, prompt)
        payload.update(init_image=_encode_png(prepared), negative_prompt=negative_prompt, strength=strength)
        try:
            result = self._server.generate(
                payload, cancellation_check=self._active_check, progress_callback=self._active_progress
            )
            return _validate_output_png(result, output_size)
        except LocalReferenceSessionCancelled as error:
            raise sdxl.StableDiffusionReferenceCancelled(str(error)) from None
        except (LocalReferenceSessionError, OSError, HTTPException) as error:
            message = str(error) if isinstance(error, LocalReferenceSessionError) else "The resident Stable Diffusion image server connection failed."
            raise sdxl.StableDiffusionReferenceError(message) from None

    def _command(self) -> list[str]:
        executable = qwen.default_qwen_model_bundle().executable.with_name(SERVER_FILENAME)
        if not executable.is_file():
            raise sdxl.StableDiffusionReferenceError(
                "The resident local image runtime (sd-server) is missing. Reinstall the bundled runtime."
            )
        model_path = sdxl.ensure_stable_diffusion_model(
            cancellation_check=self._active_check, progress_callback=self._active_progress
        )
        return [
            str(executable), "--model", str(model_path),
            "--steps", str(sdxl.INFERENCE_STEPS), "--cfg-scale", str(sdxl.GUIDANCE_SCALE),
            "--sampling-method", "dpm++2m", "--scheduler", "karras",
            "--vae-tiling", "--diffusion-fa", "--offload-to-cpu", "--mmap", "--type", "f16",
            "--disable-image-metadata",
        ]


# ### Image and networking helpers ###
def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _image_request(image: Image.Image, prompt: str) -> dict[str, object]:
    return {
        "prompt": prompt, "width": image.width, "height": image.height,
        "seed": -1, "batch_count": 1, "output_format": "png",
        "embed_image_metadata": False,
    }


def _encode_png(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG", compress_level=3)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _decode_job_output(job: dict[str, object]) -> bytes:
    try:
        images = job["result"]["images"]
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError("Expected one image")
        encoded = images[0]["b64_json"]
        if not isinstance(encoded, str) or len(encoded) > MAX_RESPONSE_BYTES:
            raise ValueError("Invalid image")
        return base64.b64decode(encoded, validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error):
        raise LocalReferenceSessionError("The local image server returned no usable image.") from None


def _validate_output_png(image_bytes: bytes, output_size: tuple[int, int]) -> bytes:
    try:
        with Image.open(BytesIO(image_bytes)) as opened:
            if opened.format != "PNG" or opened.size != output_size:
                raise LocalReferenceSessionError("The local image server returned an unexpected image size or format.")
            image = opened.convert("RGB")
        output = BytesIO()
        image.save(output, format="PNG", compress_level=3)
        return output.getvalue()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise LocalReferenceSessionError("The local image server did not return a readable PNG image.") from None


def _report(callback: ProgressCallback | None, stage: str, percent: int) -> None:
    if callback is not None:
        callback(stage, percent)
