# ### Imports ###
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# ### Constants ###
QWEN_CFG_SCALE = "6.0"
QWEN_SAMPLING_METHOD = "euler"
QWEN_DEFAULT_STEPS = 40
QWEN_SPATIAL_MULTIPLE = 32
QWEN_MAX_OUTPUT_EDGE = 2048
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_INPUT_PIXELS = 100_000_000
MAX_PROMPT_CHARACTERS = 16_000
PNG_COMPRESSION_LEVEL = 3
PROCESS_POLL_SECONDS = 0.10
PROCESS_TERMINATE_TIMEOUT_SECONDS = 5.0
QWEN_BUNDLE_DIRECTORY_NAME = "qwen-image-2.1"
QWEN_RUNTIME_DIRECTORY_NAME = "runtime"
QWEN_WEIGHTS_DIRECTORY_NAME = "weights"
QWEN_EXECUTABLE_FILENAME = "sd-cli.exe"
QWEN_DIFFUSION_FILENAME = "qwen_image_2.1-Q6_K.gguf"
QWEN_LLM_FILENAME = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
QWEN_VISION_FILENAME = "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
QWEN_VAE_FILENAME = "qwen_image_2.1_vae_bf16.safetensors"
QWEN_INFERENCE_LOCK = threading.Lock()


# ### Public types ###
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True)
class QwenSdCppModelBundle:
    """Local stable-diffusion.cpp executable and Qwen-Image-2.1 weights."""

    executable: Path
    diffusion_model: Path
    llm_model: Path
    vision_model: Path
    vae_model: Path


def default_qwen_model_directory() -> Path:
    """Return HouseMaker's per-user local Qwen model directory."""

    local_app_data = str(os.environ.get("LOCALAPPDATA", "")).strip()
    if local_app_data:
        base_directory = Path(local_app_data)
    else:
        base_directory = Path.home() / "AppData" / "Local"
    return (
        base_directory
        / "HouseMaker"
        / "models"
        / QWEN_BUNDLE_DIRECTORY_NAME
    )


def default_qwen_model_bundle() -> QwenSdCppModelBundle:
    """Build the expected downloaded stable-diffusion.cpp model bundle."""

    directory = default_qwen_model_directory()
    runtime_directory = directory / QWEN_RUNTIME_DIRECTORY_NAME
    weights_directory = directory / QWEN_WEIGHTS_DIRECTORY_NAME
    return QwenSdCppModelBundle(
        executable=runtime_directory / QWEN_EXECUTABLE_FILENAME,
        diffusion_model=weights_directory / QWEN_DIFFUSION_FILENAME,
        llm_model=weights_directory / QWEN_LLM_FILENAME,
        vision_model=weights_directory / QWEN_VISION_FILENAME,
        vae_model=weights_directory / QWEN_VAE_FILENAME,
    )


def create_default_qwen_plan_image_editor() -> QwenSdCppPlanImageEditor:
    """Create an editor from HouseMaker's downloaded local model bundle."""

    return QwenSdCppPlanImageEditor(default_qwen_model_bundle())


# ### Public exceptions ###
class QwenPlanCorrectionError(RuntimeError):
    """A local-backend error whose text is safe to show in the UI."""


class QwenPlanCorrectionCancelled(QwenPlanCorrectionError):
    pass


# ### Public backend ###
class QwenSdCppPlanImageEditor:
    """Run Qwen-Image-2.1 editing through a local CUDA sd-cli binary."""

    def __init__(
        self,
        bundle: QwenSdCppModelBundle,
        *,
        num_inference_steps: int = QWEN_DEFAULT_STEPS,
    ) -> None:
        self._bundle = _validate_bundle(bundle)
        if type(num_inference_steps) is not int or not 1 <= num_inference_steps <= 200:
            raise ValueError("Qwen inference steps must be an integer from 1 to 200.")
        self._num_inference_steps = num_inference_steps
        # One process at a time avoids loading several large model bundles into
        # system and GPU memory. Waiting callers remain cooperatively cancellable.
        self._inference_lock = QWEN_INFERENCE_LOCK

    def __call__(
        self,
        image_bytes: bytes,
        *,
        api_key: str = "",
        prompt: str,
        output_size: tuple[int, int],
        cancellation_check: CancellationCheck | None,
    ) -> bytes:
        """Edit one prepared PNG and return an exact-size PNG."""

        del api_key
        normalized_prompt = _validate_prompt(prompt)
        width, height = _validate_output_size(output_size)
        prepared_image = _decode_prepared_png(image_bytes, (width, height))
        _raise_if_cancelled(cancellation_check)
        _acquire_cancellable(self._inference_lock, cancellation_check)
        try:
            return self._run_sd_cli(
                prepared_image,
                normalized_prompt,
                (width, height),
                cancellation_check,
            )
        finally:
            self._inference_lock.release()

    def _run_sd_cli(
        self,
        prepared_image: Image.Image,
        prompt: str,
        output_size: tuple[int, int],
        cancellation_check: CancellationCheck | None,
    ) -> bytes:
        width, height = output_size
        with tempfile.TemporaryDirectory(prefix="housemaker-qwen-") as raw_directory:
            directory = Path(raw_directory)
            input_path = directory / "input.png"
            output_path = directory / "output.png"
            log_path = directory / "sd-cli.log"
            _save_input_png(input_path, prepared_image)
            command = self._build_command(
                input_path=input_path,
                output_path=output_path,
                prompt=prompt,
                width=width,
                height=height,
            )
            _raise_if_cancelled(cancellation_check)
            try:
                with log_path.open("wb") as log_file:
                    process = subprocess.Popen(
                        command,
                        cwd=str(self._bundle.executable.parent),
                        stdin=subprocess.DEVNULL,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        creationflags=_hidden_process_flags(),
                    )
                    return_code = _wait_for_process(process, cancellation_check)
            except QwenPlanCorrectionCancelled:
                raise
            except (OSError, subprocess.SubprocessError):
                raise QwenPlanCorrectionError(
                    "The local Qwen correction process could not be started."
                ) from None
            if return_code != 0:
                raise QwenPlanCorrectionError(
                    "The local Qwen correction process failed."
                )
            _raise_if_cancelled(cancellation_check)
            return _read_exact_output_png(output_path, output_size)

    def _build_command(
        self,
        *,
        input_path: Path,
        output_path: Path,
        prompt: str,
        width: int,
        height: int,
    ) -> list[str]:
        bundle = self._bundle
        return [
            str(bundle.executable),
            "--diffusion-model",
            str(bundle.diffusion_model),
            "--vae",
            str(bundle.vae_model),
            "--llm",
            str(bundle.llm_model),
            "--llm_vision",
            str(bundle.vision_model),
            "-r",
            str(input_path),
            "-p",
            prompt,
            "--cfg-scale",
            QWEN_CFG_SCALE,
            "--sampling-method",
            QWEN_SAMPLING_METHOD,
            "--steps",
            str(self._num_inference_steps),
            "--offload-to-cpu",
            "--diffusion-fa",
            "--vae-tiling",
            "-W",
            str(width),
            "-H",
            str(height),
            "-o",
            str(output_path),
        ]


# ### Bundle validation ###
def _validate_bundle(bundle: QwenSdCppModelBundle) -> QwenSdCppModelBundle:
    if not isinstance(bundle, QwenSdCppModelBundle):
        raise TypeError("Qwen model configuration must be a model bundle.")
    executable = _resolve_file(bundle.executable, "sd-cli executable")
    diffusion = _resolve_file(bundle.diffusion_model, "Qwen diffusion model")
    llm = _resolve_file(bundle.llm_model, "Qwen language model")
    vision = _resolve_file(bundle.vision_model, "Qwen vision projector")
    vae = _resolve_file(bundle.vae_model, "Qwen VAE")
    _require_filename_tokens(diffusion, ".gguf", ("q6_k",), "diffusion model")
    _require_filename_tokens(llm, ".gguf", ("q4_k_m",), "language model")
    _require_filename_tokens(
        vision,
        ".gguf",
        ("mmproj",),
        "vision projector",
    )
    vision_name = vision.name.lower()
    if not any(token in vision_name for token in ("f16", "bf16")):
        raise QwenPlanCorrectionError(
            "The Qwen vision projector must be the official F16/BF16 mmproj model."
        )
    _require_filename_tokens(vae, ".safetensors", ("bf16",), "VAE")
    return QwenSdCppModelBundle(executable, diffusion, llm, vision, vae)


def _resolve_file(raw_path: str | Path, label: str) -> Path:
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise QwenPlanCorrectionError(
            f"The configured {label} file is unavailable."
        ) from None
    if not path.is_file():
        raise QwenPlanCorrectionError(
            f"The configured {label} file is unavailable."
        )
    return path


def _require_filename_tokens(
    path: Path,
    suffix: str,
    tokens: tuple[str, ...],
    label: str,
) -> None:
    filename = path.name.lower()
    if path.suffix.lower() != suffix or any(token not in filename for token in tokens):
        raise QwenPlanCorrectionError(
            f"The configured Qwen {label} has an unsupported format or quantization."
        )


# ### Input validation ###
def _validate_prompt(prompt: str) -> str:
    normalized = str(prompt).strip()
    if not normalized:
        raise QwenPlanCorrectionError("The Qwen correction prompt is empty.")
    if len(normalized) > MAX_PROMPT_CHARACTERS:
        raise QwenPlanCorrectionError("The Qwen correction prompt is too long.")
    return normalized


def _validate_output_size(output_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(output_size, tuple) or len(output_size) != 2:
        raise QwenPlanCorrectionError("The Qwen output size is invalid.")
    width, height = output_size
    if any(type(value) is not int for value in (width, height)):
        raise QwenPlanCorrectionError("The Qwen output size is invalid.")
    if (
        min(width, height) < QWEN_SPATIAL_MULTIPLE
        or max(width, height) > QWEN_MAX_OUTPUT_EDGE
        or width % QWEN_SPATIAL_MULTIPLE
        or height % QWEN_SPATIAL_MULTIPLE
    ):
        raise QwenPlanCorrectionError(
            "Qwen output dimensions must be multiples of 32 and no larger than 2048."
        )
    return width, height


def _decode_prepared_png(
    image_bytes: bytes,
    output_size: tuple[int, int],
) -> Image.Image:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise QwenPlanCorrectionError("The prepared Qwen input is not a PNG image.")
    if len(image_bytes) > MAX_INPUT_BYTES:
        raise QwenPlanCorrectionError("The prepared Qwen input image is too large.")
    try:
        with Image.open(BytesIO(image_bytes)) as opened:
            if opened.format != "PNG":
                raise QwenPlanCorrectionError(
                    "The prepared Qwen input is not a PNG image."
                )
            if opened.width * opened.height > MAX_INPUT_PIXELS:
                raise QwenPlanCorrectionError(
                    "The prepared Qwen input image is too large."
                )
            if opened.size != output_size:
                raise QwenPlanCorrectionError(
                    "The prepared Qwen input must match the requested output size."
                )
            rgba = opened.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, rgba).convert("RGB")
    except QwenPlanCorrectionError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise QwenPlanCorrectionError(
            "The prepared Qwen input is not a readable PNG image."
        ) from None


# ### Process lifecycle ###
def _wait_for_process(
    process: subprocess.Popen[bytes],
    cancellation_check: CancellationCheck | None,
) -> int:
    while True:
        if cancellation_check is not None and bool(cancellation_check()):
            _terminate_process(process)
            raise QwenPlanCorrectionCancelled("Local Qwen correction was cancelled.")
        try:
            return int(process.wait(timeout=PROCESS_POLL_SECONDS))
        except subprocess.TimeoutExpired:
            continue


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
        process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=PROCESS_TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _hidden_process_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))


# ### PNG input and output ###
def _save_input_png(path: Path, image: Image.Image) -> None:
    try:
        image.save(path, format="PNG", compress_level=PNG_COMPRESSION_LEVEL)
    except OSError:
        raise QwenPlanCorrectionError(
            "The prepared Qwen input could not be staged."
        ) from None


def _read_exact_output_png(path: Path, output_size: tuple[int, int]) -> bytes:
    try:
        with Image.open(path) as opened:
            if opened.format != "PNG" or opened.size != output_size:
                raise QwenPlanCorrectionError(
                    "Local Qwen correction returned an unexpected image size."
                )
            opened.load()
            image = opened.convert("RGB")
    except QwenPlanCorrectionError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        raise QwenPlanCorrectionError(
            "Local Qwen correction did not return a readable PNG image."
        ) from None
    output = BytesIO()
    try:
        image.save(output, format="PNG", compress_level=PNG_COMPRESSION_LEVEL)
    except OSError:
        raise QwenPlanCorrectionError(
            "The local Qwen result could not be encoded as PNG."
        ) from None
    return output.getvalue()


# ### Cancellation helpers ###
def _acquire_cancellable(
    lock: threading.Lock,
    cancellation_check: CancellationCheck | None,
) -> None:
    while not lock.acquire(timeout=PROCESS_POLL_SECONDS):
        _raise_if_cancelled(cancellation_check)
    try:
        _raise_if_cancelled(cancellation_check)
    except Exception:
        lock.release()
        raise


def _raise_if_cancelled(cancellation_check: CancellationCheck | None) -> None:
    if cancellation_check is not None and bool(cancellation_check()):
        raise QwenPlanCorrectionCancelled("Local Qwen correction was cancelled.")
