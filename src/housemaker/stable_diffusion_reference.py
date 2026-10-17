# ### Imports ###
from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from .qwen_plan_image_correction import QWEN_INFERENCE_LOCK, default_qwen_model_bundle
from .reference_generation_models import DEFAULT_REFERENCE_NEGATIVE_PROMPT

# ### Constants ###
MODEL_FILENAME = "sd_xl_base_1.0.safetensors"
MODEL_SIZE_BYTES = 6_938_078_334
MODEL_SHA256 = "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
MODEL_REVISION = "ce8cd4d6569112b0c0ad48cdd18560fe0bcc7a48"
MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/"
    f"{MODEL_REVISION}/{MODEL_FILENAME}"
)
MODEL_DOWNLOAD_LABEL = "Downloading Stable Diffusion XL (6.94 GB, first use only)"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 10.0
DOWNLOAD_DISK_MARGIN_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_PROMPT_CHARACTERS = 16_000
PROCESS_POLL_SECONDS = 0.10
PROCESS_TERMINATE_TIMEOUT_SECONDS = 5.0
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 6.0
NEGATIVE_PROMPT = DEFAULT_REFERENCE_NEGATIVE_PROMPT
_MODEL_DOWNLOAD_LOCK = threading.Lock()


# ### Public types and exceptions ###
CancellationCheck = Callable[[], bool]
ProgressCallback = Callable[[str, int], None]


class StableDiffusionReferenceError(RuntimeError):
    """A local img2img failure that is safe to display to the user."""


class StableDiffusionReferenceCancelled(StableDiffusionReferenceError):
    pass


# ### Public backend ###
def default_stable_diffusion_model_directory() -> Path:
    """Keep the optional SDXL download outside the repository and project files."""

    return default_qwen_model_bundle().executable.parents[2] / "stable-diffusion-xl"


def generate_stable_diffusion_reference(
    image_bytes: bytes,
    *,
    prompt: str,
    negative_prompt: str = DEFAULT_REFERENCE_NEGATIVE_PROMPT,
    output_size: tuple[int, int] = (1024, 1024),
    strength: float = 0.45,
    cancellation_check: CancellationCheck | None = None,
    progress_callback: ProgressCallback | None = None,
) -> bytes:
    """Regenerate a prepared object reference locally using SDXL img2img.

    Nothing is downloaded or loaded at application startup. The first local
    generation downloads the official full checkpoint, including its encoders
    and VAE. All generations share the existing local-Qwen GPU mutex.
    """

    prepared_image, normalized_prompt, normalized_strength = _validate_inputs(
        image_bytes, prompt, output_size, strength
    )
    _validate_negative_prompt(negative_prompt)
    _raise_if_cancelled(cancellation_check)
    executable = default_qwen_model_bundle().executable
    if not executable.is_file():
        raise StableDiffusionReferenceError(
            "The local Stable Diffusion runtime is missing. Install HouseMaker's "
            "local Qwen/Stable Diffusion runtime before generating references."
        )
    model_path = ensure_stable_diffusion_model(
        cancellation_check=cancellation_check,
        progress_callback=progress_callback,
    )
    _report(progress_callback, "Waiting for the local image-generation GPU", 0)
    _acquire_cancellable(QWEN_INFERENCE_LOCK, cancellation_check)
    try:
        return _run_img2img(
            executable,
            model_path,
            prepared_image,
            normalized_prompt,
            negative_prompt,
            normalized_strength,
            cancellation_check,
            progress_callback,
        )
    finally:
        QWEN_INFERENCE_LOCK.release()


def ensure_stable_diffusion_model(
    *,
    cancellation_check: CancellationCheck | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Download and checksum the pinned official checkpoint once, atomically."""

    model_path = default_stable_diffusion_model_directory() / MODEL_FILENAME
    _raise_if_cancelled(cancellation_check)
    _acquire_cancellable(_MODEL_DOWNLOAD_LOCK, cancellation_check)
    try:
        if _is_complete_checkpoint(model_path):
            return model_path
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(model_path.parent).free
        except OSError:
            raise StableDiffusionReferenceError(
                "The local Stable Diffusion model cache is not writable."
            ) from None
        if free_bytes < MODEL_SIZE_BYTES + DOWNLOAD_DISK_MARGIN_BYTES:
            raise StableDiffusionReferenceError(
                "Downloading Stable Diffusion XL requires at least 7 GB of free disk space."
            )
        _download_checkpoint(model_path, cancellation_check, progress_callback)
        return model_path
    finally:
        _MODEL_DOWNLOAD_LOCK.release()


# ### Input validation ###
def _validate_inputs(
    image_bytes: bytes,
    prompt: str,
    output_size: tuple[int, int],
    strength: float,
) -> tuple[Image.Image, str, float]:
    if (
        not isinstance(output_size, tuple)
        or len(output_size) != 2
        or any(type(value) is not int for value in output_size)
        or any(value < 256 or value > 1536 or value % 64 for value in output_size)
    ):
        raise StableDiffusionReferenceError(
            "Stable Diffusion output dimensions must be multiples of 64 between 256 and 1536."
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise StableDiffusionReferenceError("Enter a reference-generation prompt first.")
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise StableDiffusionReferenceError("The reference-generation prompt is too long.")
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(strength)
        or not 0 < strength <= 1
    ):
        raise StableDiffusionReferenceError("Img2img strength must be greater than 0 and at most 1.")
    if not isinstance(image_bytes, bytes) or not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise StableDiffusionReferenceError("The prepared object reference is not a usable PNG image.")
    try:
        with Image.open(BytesIO(image_bytes)) as opened:
            if opened.format != "PNG" or opened.size != output_size:
                raise StableDiffusionReferenceError(
                    "The prepared object reference must be a PNG matching the requested output size."
                )
            rgba = opened.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            prepared = Image.alpha_composite(background, rgba).convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise StableDiffusionReferenceError("The prepared object reference is not readable.") from None
    return prepared, prompt.strip(), float(strength)


def _validate_negative_prompt(negative_prompt: str) -> None:
    if not isinstance(negative_prompt, str):
        raise StableDiffusionReferenceError("The reference-generation negative prompt must be text.")
    if len(negative_prompt) > MAX_PROMPT_CHARACTERS:
        raise StableDiffusionReferenceError("The reference-generation negative prompt is too long.")
    if "\x00" in negative_prompt:
        raise StableDiffusionReferenceError("The reference-generation negative prompt contains an invalid character.")


# ### Model download and cache ###
def _is_complete_checkpoint(path: Path) -> bool:
    try:
        if path.stat().st_size != MODEL_SIZE_BYTES:
            return False
        with path.open("rb") as checkpoint:
            header_size = int.from_bytes(checkpoint.read(8), "little")
        return 0 < header_size < DOWNLOAD_DISK_MARGIN_BYTES
    except OSError:
        return False


def _download_checkpoint(
    destination: Path,
    cancellation_check: CancellationCheck | None,
    progress_callback: ProgressCallback | None,
) -> None:
    partial_path: Path | None = None
    _report(progress_callback, MODEL_DOWNLOAD_LABEL, 0)
    try:
        request = Request(MODEL_DOWNLOAD_URL, headers={"User-Agent": "HouseMaker/0.1"})
        with (
            urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{MODEL_FILENAME}.", suffix=".part", dir=destination.parent, delete=False
            ) as partial,
        ):
            partial_path = Path(partial.name)
            digest = hashlib.sha256()
            downloaded_bytes = 0
            last_percent = -1
            while True:
                _raise_if_cancelled(cancellation_check)
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                downloaded_bytes += len(chunk)
                if downloaded_bytes > MODEL_SIZE_BYTES:
                    raise StableDiffusionReferenceError(
                        "The Stable Diffusion download was larger than the official checkpoint."
                    )
                partial.write(chunk)
                digest.update(chunk)
                percent = int(downloaded_bytes * 100 / MODEL_SIZE_BYTES)
                if percent != last_percent:
                    _report(progress_callback, MODEL_DOWNLOAD_LABEL, percent)
                    last_percent = percent
        _raise_if_cancelled(cancellation_check)
        if downloaded_bytes != MODEL_SIZE_BYTES or digest.hexdigest() != MODEL_SHA256:
            raise StableDiffusionReferenceError(
                "The Stable Diffusion model download was incomplete or failed its checksum. Retry the generation."
            )
        os.replace(partial_path, destination)
        partial_path = None
    except HTTPError as error:
        raise StableDiffusionReferenceError(
            f"The official Stable Diffusion model download failed (HTTP {error.code})."
        ) from None
    except (URLError, TimeoutError):
        raise StableDiffusionReferenceError(
            "The Stable Diffusion model download could not reach Hugging Face or timed out."
        ) from None
    except OSError:
        raise StableDiffusionReferenceError(
            "The Stable Diffusion model could not be downloaded or saved. Check network access and disk space."
        ) from None
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)


# ### Img2img process ###
def _run_img2img(
    executable: Path,
    model_path: Path,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    cancellation_check: CancellationCheck | None,
    progress_callback: ProgressCallback | None,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="housemaker-sdxl-reference-") as raw_directory:
        directory = Path(raw_directory)
        input_path = directory / "reference.png"
        output_path = directory / "generated.png"
        prompt_path = directory / "prompt.txt"
        log_path = directory / "sd-cli.log"
        try:
            image.save(input_path, format="PNG", compress_level=3)
            prompt_path.write_text(prompt, encoding="utf-8")
            command = _build_command(
                executable, model_path, input_path, output_path, prompt_path,
                image.size, strength, negative_prompt,
            )
            _raise_if_cancelled(cancellation_check)
            _report(progress_callback, "Regenerating the object reference with Stable Diffusion XL", 0)
            with log_path.open("wb") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(executable.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)) if os.name == "nt" else 0,
                )
                try:
                    while True:
                        _raise_if_cancelled(cancellation_check)
                        try:
                            return_code = process.wait(timeout=PROCESS_POLL_SECONDS)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                finally:
                    if process.poll() is None:
                        _terminate_process(process)
            if return_code:
                raise StableDiffusionReferenceError(_process_error_message(log_path, int(return_code)))
            _raise_if_cancelled(cancellation_check)
            result = _read_output(output_path, image.size)
            _report(progress_callback, "Reference generation complete", 100)
            return result
        except StableDiffusionReferenceError:
            raise
        except (OSError, subprocess.SubprocessError):
            raise StableDiffusionReferenceError(
                "The local Stable Diffusion process could not start or save its reference image."
            ) from None


def _build_command(
    executable: Path,
    model_path: Path,
    input_path: Path,
    output_path: Path,
    prompt_path: Path,
    output_size: tuple[int, int],
    strength: float,
    negative_prompt: str = DEFAULT_REFERENCE_NEGATIVE_PROMPT,
) -> list[str]:
    return [
        str(executable), "--model", str(model_path),
        "--init-img", str(input_path), "--prompt-file", str(prompt_path),
        "--negative-prompt", negative_prompt, "--strength", str(strength),
        "--steps", str(INFERENCE_STEPS), "--cfg-scale", str(GUIDANCE_SCALE),
        "--sampling-method", "dpm++2m", "--scheduler", "karras",
        "--width", str(output_size[0]), "--height", str(output_size[1]),
        "--seed", "-1", "--batch-count", "1", "--vae-tiling",
        "--diffusion-fa", "--offload-to-cpu", "--mmap", "--type", "f16",
        "--disable-image-metadata", "--output", str(output_path),
    ]


def _process_error_message(log_path: Path, return_code: int) -> str:
    # Do not leak the reference prompt or private paths echoed by the CLI. Known
    # failure categories still give the user a useful reason instead of a fallback.
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(max(0, log_path.stat().st_size - 16_384))
            log = log_file.read().decode("utf-8", errors="replace").lower()
    except OSError:
        log = ""
    reasons = (
        (("out of memory", "failed to allocate", "cuda_error_out_of_memory"), "insufficient GPU or system memory"),
        (("unknown argument", "unknown option", "unrecognized option"), "the installed runtime does not support the required img2img options"),
        (("failed to load model", "unknown model", "unsupported model", "invalid model"), "the runtime could not load the SDXL checkpoint"),
        (("no cuda-capable", "cuda driver", "cuda error", "cudaerror"), "the CUDA runtime or GPU is unavailable"),
    )
    for needles, reason in reasons:
        if any(needle in log for needle in needles):
            return f"Stable Diffusion reference generation failed: {reason} (exit {return_code})."
    return f"The local Stable Diffusion img2img process failed (exit {return_code})."


def _read_output(path: Path, expected_size: tuple[int, int]) -> bytes:
    try:
        with Image.open(path) as opened:
            if opened.format != "PNG" or opened.size != expected_size:
                raise StableDiffusionReferenceError("Stable Diffusion returned an unexpected image size or format.")
            image = opened.convert("RGB")
        output = BytesIO()
        image.save(output, format="PNG", compress_level=3)
        return output.getvalue()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise StableDiffusionReferenceError("Stable Diffusion did not return a readable reference image.") from None


# ### Cooperative cancellation and progress ###
def _raise_if_cancelled(cancellation_check: CancellationCheck | None) -> None:
    if cancellation_check is not None and cancellation_check():
        raise StableDiffusionReferenceCancelled("Stable Diffusion reference generation was cancelled.")


def _acquire_cancellable(lock: threading.Lock, cancellation_check: CancellationCheck | None) -> None:
    while not lock.acquire(timeout=PROCESS_POLL_SECONDS):
        _raise_if_cancelled(cancellation_check)
    try:
        _raise_if_cancelled(cancellation_check)
    except BaseException:
        lock.release()
        raise


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
        process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _report(progress_callback: ProgressCallback | None, message: str, percent: int) -> None:
    if progress_callback is not None:
        progress_callback(message, percent)
