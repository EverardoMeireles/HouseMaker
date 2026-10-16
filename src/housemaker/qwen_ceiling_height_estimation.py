# ### Imports ###
from __future__ import annotations

import json
import os
import queue
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from housemaker.qwen_plan_image_correction import (
    QWEN_INFERENCE_LOCK,
    QWEN_LLM_FILENAME,
    QWEN_VISION_FILENAME,
    default_qwen_model_directory,
)

# ### Constants ###
LOCAL_QWEN_CEILING_HEIGHT_MODEL = "qwen3-vl-8b-local"
QWEN_LLAMA_RUNTIME_DIRECTORY_NAME = "llama-runtime"
QWEN_LLAMA_SERVER_FILENAME = "llama-server.exe"
QWEN_CUDA_RUNTIME_DIRECTORY_NAME = "runtime"
QWEN_WEIGHTS_DIRECTORY_NAME = "weights"
QWEN_SERVER_HOST = "127.0.0.1"
QWEN_SERVER_CONTEXT_TOKENS = 8_192
QWEN_SERVER_IMAGE_MIN_TOKENS = 1_024
QWEN_SERVER_IMAGE_MAX_TOKENS = 2_048
QWEN_SERVER_START_TIMEOUT_SECONDS = 120.0
QWEN_REQUEST_TIMEOUT_SECONDS = 180.0
QWEN_PROCESS_POLL_SECONDS = 0.10
QWEN_HEALTH_REQUEST_TIMEOUT_SECONDS = 1.0
QWEN_PROCESS_TERMINATE_TIMEOUT_SECONDS = 5.0
QWEN_MAX_RESPONSE_BYTES = 1 * 1024 * 1024
QWEN_MAX_OUTPUT_TOKENS = 500


# ### Public types ###
CancellationCheck = Callable[[], bool]


class UrlOpenFunction(Protocol):
    def __call__(self, request: Request, **kwargs: Any) -> object: ...


@dataclass(frozen=True)
class QwenCeilingHeightModelBundle:
    """Files required by the local Qwen vision-language server."""

    executable: Path
    language_model: Path
    vision_projector: Path
    cuda_runtime_directory: Path


# ### Public exceptions ###
class QwenCeilingHeightError(RuntimeError):
    """A local-backend failure whose text is safe to display in the UI."""


class QwenCeilingHeightCancelled(QwenCeilingHeightError):
    pass


# ### Default bundle ###
def default_qwen_ceiling_height_bundle() -> QwenCeilingHeightModelBundle:
    """Return the local llama.cpp runtime and existing Qwen3-VL weights."""

    directory = default_qwen_model_directory()
    return QwenCeilingHeightModelBundle(
        executable=(
            directory
            / QWEN_LLAMA_RUNTIME_DIRECTORY_NAME
            / QWEN_LLAMA_SERVER_FILENAME
        ),
        language_model=(
            directory / QWEN_WEIGHTS_DIRECTORY_NAME / QWEN_LLM_FILENAME
        ),
        vision_projector=(
            directory / QWEN_WEIGHTS_DIRECTORY_NAME / QWEN_VISION_FILENAME
        ),
        cuda_runtime_directory=(
            directory / QWEN_CUDA_RUNTIME_DIRECTORY_NAME
        ),
    )


# ### Local server session ###
class QwenCeilingHeightServer:
    """Load Qwen once and serve every frame in one analysis job."""

    def __init__(
        self,
        bundle: QwenCeilingHeightModelBundle | None = None,
        *,
        cancellation_check: CancellationCheck | None = None,
        opener: UrlOpenFunction | None = None,
        context_tokens: int = QWEN_SERVER_CONTEXT_TOKENS,
    ) -> None:
        self._bundle = _validate_bundle(
            default_qwen_ceiling_height_bundle() if bundle is None else bundle
        )
        self._startup_cancellation_check = cancellation_check
        self._opener = urlopen if opener is None else opener
        self._context_tokens = max(QWEN_SERVER_CONTEXT_TOKENS, int(context_tokens))
        self._process: subprocess.Popen[bytes] | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._log_file: object | None = None
        self._port: int | None = None
        self._api_key = ""
        self._owns_inference_lock = False

    def __enter__(self) -> QwenCeilingHeightServer:  # noqa: PYI034
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def start(self) -> None:
        """Acquire the shared Qwen GPU lock and start a private loopback server."""

        if self._process is not None:
            return
        _acquire_cancellable(
            QWEN_INFERENCE_LOCK,
            self._startup_cancellation_check,
        )
        self._owns_inference_lock = True
        try:
            self._raise_if_cancelled(self._startup_cancellation_check)
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="housemaker-qwen-ceiling-"
            )
            log_path = Path(self._temporary_directory.name) / "llama-server.log"
            self._log_file = log_path.open("wb")
            self._port = _reserve_loopback_port()
            self._api_key = secrets.token_urlsafe(32)
            self._process = subprocess.Popen(
                self._build_command(),
                cwd=str(self._bundle.executable.parent),
                env=self._build_environment(),
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=_hidden_process_flags(),
            )
            self._wait_until_ready()
        except QwenCeilingHeightCancelled:
            self.close()
            raise
        except QwenCeilingHeightError:
            self.close()
            raise
        # This boundary owns the process and lock, so every startup failure
        # must release them before returning control to the caller.
        except Exception:  # noqa: BLE001
            self.close()
            raise QwenCeilingHeightError(
                "The local Qwen ceiling-height server could not be started."
            ) from None

    def close(self) -> None:
        """Stop the private server, release VRAM, and release the shared lock."""

        process = self._process
        self._process = None
        if process is not None:
            _terminate_process(process)
        log_file = self._log_file
        self._log_file = None
        if log_file is not None:
            try:
                log_file.close()  # type: ignore[attr-defined]
            except OSError:
                pass
        temporary_directory = self._temporary_directory
        self._temporary_directory = None
        if temporary_directory is not None:
            try:
                temporary_directory.cleanup()
            except OSError:
                pass
        self._port = None
        self._api_key = ""
        if self._owns_inference_lock:
            self._owns_inference_lock = False
            QWEN_INFERENCE_LOCK.release()

    def estimate(
        self,
        image_data_uri: str,
        *,
        previous_image_data_uri: str | None = None,
        plan_image_data_uri: str | None = None,
        additional_images: Sequence[tuple[str, str]] = (),
        max_output_tokens: int = QWEN_MAX_OUTPUT_TOKENS,
        prompt: str,
        result_schema: Mapping[str, Any],
        cancellation_check: CancellationCheck | None = None,
    ) -> object:
        """Return one schema-constrained result payload for an encoded frame."""

        process = self._process
        if process is None or self._port is None or not self._api_key:
            raise QwenCeilingHeightError(
                "The local Qwen ceiling-height server is not running."
            )
        poll = getattr(process, "poll", None)
        if callable(poll) and poll() is not None:
            raise QwenCeilingHeightError(
                "The local Qwen ceiling-height server stopped unexpectedly."
            )
        self._raise_if_cancelled(cancellation_check)
        request_body = build_local_qwen_request_body(
            image_data_uri,
            previous_image_data_uri=previous_image_data_uri,
            plan_image_data_uri=plan_image_data_uri,
            additional_images=additional_images,
            max_output_tokens=max_output_tokens,
            prompt=prompt,
            result_schema=result_schema,
        )
        request = Request(
            f"http://{QWEN_SERVER_HOST}:{self._port}/v1/chat/completions",
            data=json.dumps(request_body, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "HouseMaker/1.0",
            },
        )
        return self._run_request_cancellably(request, cancellation_check)

    # ### Server construction ###
    def _build_command(self) -> list[str]:
        assert self._port is not None
        return [
            str(self._bundle.executable),
            "--model",
            str(self._bundle.language_model),
            "--mmproj",
            str(self._bundle.vision_projector),
            "--host",
            QWEN_SERVER_HOST,
            "--port",
            str(self._port),
            "--ctx-size",
            str(self._context_tokens),
            "--n-gpu-layers",
            "all",
            "--fit",
            "on",
            "--flash-attn",
            "on",
            "--parallel",
            "1",
            "--jinja",
            "--image-min-tokens",
            str(QWEN_SERVER_IMAGE_MIN_TOKENS),
            "--image-max-tokens",
            str(QWEN_SERVER_IMAGE_MAX_TOKENS),
            "--no-webui",
            "--api-key",
            self._api_key,
        ]

    def _build_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        search_paths = (
            str(self._bundle.executable.parent),
            str(self._bundle.cuda_runtime_directory),
            environment.get("PATH", ""),
        )
        environment["PATH"] = os.pathsep.join(
            path for path in search_paths if path
        )
        return environment

    def _wait_until_ready(self) -> None:
        assert self._port is not None
        deadline = time.monotonic() + QWEN_SERVER_START_TIMEOUT_SECONDS
        health_request = Request(
            f"http://{QWEN_SERVER_HOST}:{self._port}/health",
            method="GET",
            headers={"Accept": "application/json"},
        )
        while time.monotonic() < deadline:
            self._raise_if_cancelled(self._startup_cancellation_check)
            process = self._process
            if process is None or process.poll() is not None:
                raise QwenCeilingHeightError(
                    "The local Qwen model could not be loaded."
                )
            try:
                with self._opener(
                    health_request,
                    timeout=QWEN_HEALTH_REQUEST_TIMEOUT_SECONDS,
                ) as response:
                    payload = json.loads(
                        _read_response_limited(response).decode("utf-8")
                    )
                if isinstance(payload, Mapping) and payload.get("status") == "ok":
                    return
            except (HTTPError, URLError, OSError, TimeoutError, ValueError):
                pass
            time.sleep(QWEN_PROCESS_POLL_SECONDS)
        raise QwenCeilingHeightError(
            "The local Qwen model took too long to load."
        )

    # ### Request lifecycle ###
    def _run_request_cancellably(
        self,
        request: Request,
        cancellation_check: CancellationCheck | None,
    ) -> object:
        outcome: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                with self._opener(
                    request,
                    timeout=QWEN_REQUEST_TIMEOUT_SECONDS,
                ) as response:
                    raw_response = _read_response_limited(response)
                payload = json.loads(raw_response.decode("utf-8"))
                result = extract_local_qwen_result(payload)
            except Exception as error:  # noqa: BLE001 - transferred to caller.
                outcome.put((False, error))
            else:
                outcome.put((True, result))

        threading.Thread(
            target=invoke,
            name="ceiling-height-qwen-request",
            daemon=True,
        ).start()
        while True:
            if cancellation_check is not None and cancellation_check():
                self.close()
                raise QwenCeilingHeightCancelled(
                    "Ceiling height inference was cancelled."
                )
            try:
                succeeded, value = outcome.get(timeout=QWEN_PROCESS_POLL_SECONDS)
            except queue.Empty:
                continue
            if succeeded:
                return value
            if isinstance(value, QwenCeilingHeightError):
                raise value
            raise QwenCeilingHeightError(
                "The local Qwen model did not return a usable ceiling-height estimate."
            ) from None

    def _raise_if_cancelled(
        self,
        cancellation_check: CancellationCheck | None,
    ) -> None:
        if cancellation_check is not None and cancellation_check():
            raise QwenCeilingHeightCancelled(
                "Ceiling height inference was cancelled."
            )


# ### Request construction ###
def build_local_qwen_request_body(
    image_data_uri: str,
    *,
    previous_image_data_uri: str | None = None,
    plan_image_data_uri: str | None = None,
    additional_images: Sequence[tuple[str, str]] = (),
    max_output_tokens: int = QWEN_MAX_OUTPUT_TOKENS,
    prompt: str,
    result_schema: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_uri = str(image_data_uri).strip()
    normalized_previous_uri = (
        None
        if previous_image_data_uri is None
        else str(previous_image_data_uri).strip()
    )
    normalized_plan_uri = (
        None
        if plan_image_data_uri is None
        else str(plan_image_data_uri).strip()
    )
    normalized_prompt = str(prompt).strip()
    if not normalized_uri.startswith("data:image/"):
        raise QwenCeilingHeightError(
            "The current video frame could not be prepared for local analysis."
        )
    if not normalized_prompt:
        raise QwenCeilingHeightError(
            "The local ceiling-height prompt is empty."
        )
    if normalized_previous_uri is not None and not normalized_previous_uri.startswith(
        "data:image/"
    ):
        raise QwenCeilingHeightError(
            "The previous video frame could not be prepared for local analysis."
        )
    if normalized_plan_uri is not None and not normalized_plan_uri.startswith(
        "data:image/"
    ):
        raise QwenCeilingHeightError(
            "The metric plan reference could not be prepared for local analysis."
        )
    content: list[dict[str, object]] = [
        {"type": "text", "text": normalized_prompt}
    ]
    if normalized_plan_uri is not None:
        content.extend(
            (
                {
                    "type": "text",
                    "text": (
                        "PLAN REFERENCE for the current level; M labels match "
                        "the calibrated horizontal spans in the prompt:"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": normalized_plan_uri},
                },
            )
        )
    if normalized_previous_uri is not None:
        content.extend(
            (
                {"type": "text", "text": "PREVIOUS sampled frame:"},
                {
                    "type": "image_url",
                    "image_url": {"url": normalized_previous_uri},
                },
            )
        )
    content.extend(
        (
            {"type": "text", "text": "CURRENT sampled frame:"},
            {
                "type": "image_url",
                "image_url": {"url": normalized_uri},
            },
        )
    )
    for label, image_uri in additional_images:
        if not str(label).strip() or not str(image_uri).startswith("data:image/"):
            raise QwenCeilingHeightError(
                "An additional reference image could not be prepared for local analysis."
            )
        content.extend(
            (
                {"type": "text", "text": str(label)},
                {"type": "image_url", "image_url": {"url": str(image_uri)}},
            )
        )
    return {
        "model": LOCAL_QWEN_CEILING_HEIGHT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": 0,
        "max_tokens": max(1, int(max_output_tokens)),
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "ceiling_height_estimate",
                "strict": True,
                "schema": dict(result_schema),
            },
        },
    }


# ### Response parsing ###
def extract_local_qwen_result(payload: object) -> object:
    if not isinstance(payload, Mapping):
        raise QwenCeilingHeightError(
            "The local Qwen model returned an invalid response."
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise QwenCeilingHeightError(
            "The local Qwen model completed without an estimate."
        )
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise QwenCeilingHeightError(
            "The local Qwen model completed without an estimate."
        )
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise QwenCeilingHeightError(
            "The local Qwen model returned an invalid estimate."
        ) from None


# ### Bundle validation ###
def _validate_bundle(
    bundle: QwenCeilingHeightModelBundle,
) -> QwenCeilingHeightModelBundle:
    if not isinstance(bundle, QwenCeilingHeightModelBundle):
        raise TypeError("Qwen ceiling-height configuration must be a model bundle.")
    executable = _resolve_file(bundle.executable, "llama.cpp server")
    language_model = _resolve_file(bundle.language_model, "Qwen language model")
    vision_projector = _resolve_file(
        bundle.vision_projector,
        "Qwen vision projector",
    )
    cuda_runtime = _resolve_directory(
        bundle.cuda_runtime_directory,
        "Qwen CUDA runtime",
    )
    _require_filename_tokens(
        executable,
        ".exe",
        ("llama", "server"),
        "server",
    )
    _require_filename_tokens(
        language_model,
        ".gguf",
        ("qwen3vl", "8b", "q4_k_m"),
        "language model",
    )
    _require_filename_tokens(
        vision_projector,
        ".gguf",
        ("mmproj", "qwen3vl", "8b"),
        "vision projector",
    )
    return QwenCeilingHeightModelBundle(
        executable,
        language_model,
        vision_projector,
        cuda_runtime,
    )


def _resolve_file(raw_path: str | Path, label: str) -> Path:
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise QwenCeilingHeightError(
            f"The configured {label} file is unavailable."
        ) from None
    if not path.is_file():
        raise QwenCeilingHeightError(
            f"The configured {label} file is unavailable."
        )
    return path


def _resolve_directory(raw_path: str | Path, label: str) -> Path:
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise QwenCeilingHeightError(
            f"The configured {label} directory is unavailable."
        ) from None
    if not path.is_dir():
        raise QwenCeilingHeightError(
            f"The configured {label} directory is unavailable."
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
        raise QwenCeilingHeightError(
            f"The configured Qwen {label} has an unsupported format."
        )


# ### Process helpers ###
def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((QWEN_SERVER_HOST, 0))
        return int(server_socket.getsockname()[1])


def _acquire_cancellable(
    lock: threading.Lock,
    cancellation_check: CancellationCheck | None,
) -> None:
    while not lock.acquire(timeout=QWEN_PROCESS_POLL_SECONDS):
        if cancellation_check is not None and cancellation_check():
            raise QwenCeilingHeightCancelled(
                "Ceiling height inference was cancelled."
            )
    if cancellation_check is not None and cancellation_check():
        lock.release()
        raise QwenCeilingHeightCancelled(
            "Ceiling height inference was cancelled."
        )


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=QWEN_PROCESS_TERMINATE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=QWEN_PROCESS_TERMINATE_TIMEOUT_SECONDS)
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


# ### HTTP helpers ###
def _read_response_limited(response: object) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(64 * 1024)  # type: ignore[attr-defined]
        if not chunk:
            break
        total += len(chunk)
        if total > QWEN_MAX_RESPONSE_BYTES:
            raise QwenCeilingHeightError(
                "The local Qwen model returned an unexpectedly large response."
            )
        chunks.append(chunk)
    return b"".join(chunks)


__all__ = [
    "LOCAL_QWEN_CEILING_HEIGHT_MODEL",
    "QwenCeilingHeightCancelled",
    "QwenCeilingHeightError",
    "QwenCeilingHeightModelBundle",
    "QwenCeilingHeightServer",
    "build_local_qwen_request_body",
    "default_qwen_ceiling_height_bundle",
    "extract_local_qwen_result",
]
