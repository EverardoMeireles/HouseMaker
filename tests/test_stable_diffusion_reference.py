# ### Imports ###
from __future__ import annotations

import hashlib
import subprocess
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from PIL import Image

from housemaker import stable_diffusion_reference as backend
from housemaker.reference_generation_models import DEFAULT_REFERENCE_NEGATIVE_PROMPT


# ### Fixture helpers ###
def _png_bytes(size: tuple[int, int] = (1024, 1024)) -> bytes:
    output = BytesIO()
    Image.new("RGBA", size, (40, 80, 100, 128)).save(output, format="PNG")
    return output.getvalue()


def _checkpoint_bytes() -> bytes:
    header = b'{"__metadata__":{}}'
    return len(header).to_bytes(8, "little") + header + b"model fixture"


# ### Process doubles ###
class FakeProcess:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.completed = False
        self.terminated = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        self.completed = True
        return self.return_code

    def poll(self) -> int | None:
        return self.return_code if self.completed else None

    def terminate(self) -> None:
        self.terminated = True
        self.completed = True

    def kill(self) -> None:
        self.killed = True
        self.completed = True


class HangingProcess(FakeProcess):
    def __init__(self, cancellation: dict[str, bool], *, require_kill: bool = False) -> None:
        super().__init__()
        self.cancellation = cancellation
        self.require_kill = require_kill

    def wait(self, timeout: float | None = None) -> int:
        if self.killed or (self.terminated and not self.require_kill):
            self.completed = True
            return -1
        self.cancellation["cancelled"] = True
        raise subprocess.TimeoutExpired("sd-cli", timeout)

    def poll(self) -> int | None:
        return -1 if self.killed or (self.terminated and not self.require_kill) else None


# ### Backend tests ###
class StableDiffusionReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.executable = self.directory / "runtime" / "sd-cli.exe"
        self.executable.parent.mkdir()
        self.executable.write_bytes(b"runtime fixture")
        self.model_path = self.directory / backend.MODEL_FILENAME
        self.model_path.write_bytes(b"model fixture")
        self.runtime_patch = patch.object(
            backend, "default_qwen_model_bundle", return_value=SimpleNamespace(executable=self.executable)
        )
        self.ensure_patch = patch.object(backend, "ensure_stable_diffusion_model", return_value=self.model_path)
        self.runtime_patch.start()
        self.ensure_model = self.ensure_patch.start()
        self.lock = threading.Lock()
        self.lock_patch = patch.object(backend, "QWEN_INFERENCE_LOCK", self.lock)
        self.lock_patch.start()

    def tearDown(self) -> None:
        self.lock_patch.stop()
        self.ensure_patch.stop()
        self.runtime_patch.stop()
        self.temporary.cleanup()

    def test_uses_img2img_prompt_file_and_returns_opaque_exact_png(self) -> None:
        calls = []
        prompt = 'a table, "no cloth"; $(not-a-command)\nwhite background'

        def start(command: list[str], **kwargs: object) -> FakeProcess:
            calls.append((command, kwargs))
            prompt_path = Path(command[command.index("--prompt-file") + 1])
            self.assertEqual(prompt_path.read_text(encoding="utf-8"), prompt)
            self.assertNotIn(prompt, command)
            source_path = Path(command[command.index("--init-img") + 1])
            with Image.open(source_path) as source:
                self.assertEqual(source.mode, "RGB")
                self.assertEqual(source.size, (1024, 1024))
            Path(command[command.index("--output") + 1]).write_bytes(_png_bytes())
            return FakeProcess()

        progress = []
        with patch.object(backend.subprocess, "Popen", side_effect=start):
            result = backend.generate_stable_diffusion_reference(
                _png_bytes(), prompt=prompt, strength=0.4, progress_callback=lambda *args: progress.append(args)
            )
        with Image.open(BytesIO(result)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (1024, 1024))
        command, kwargs = calls[0]
        self.assertEqual(command[command.index("--model") + 1], str(self.model_path))
        self.assertEqual(command[command.index("--strength") + 1], "0.4")
        self.assertEqual(command[command.index("--steps") + 1], "30")
        self.assertEqual(command[command.index("--width") + 1], "1024")
        self.assertEqual(command[command.index("--negative-prompt") + 1], DEFAULT_REFERENCE_NEGATIVE_PROMPT)
        self.assertIn("--offload-to-cpu", command)
        self.assertIn("--vae-tiling", command)
        self.assertIn("--disable-image-metadata", command)
        self.assertNotIn("--ref-image", command)
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(progress[-1], ("Reference generation complete", 100))
        self.assertFalse(self.lock.locked())

    def test_custom_and_empty_negative_prompts_are_passed_as_one_unchanged_argument(self) -> None:
        negative_prompts = (
            'perspective view, "painted boundary"; $(not-a-command)\nblurry café cupboard',
            "",
            "  mask outline, background  ",
        )
        for negative_prompt in negative_prompts:
            commands: list[list[str]] = []

            def start(
                command: list[str],
                *,
                expected_negative_prompt: str = negative_prompt,
                recorded_commands: list[list[str]] = commands,
                **kwargs: object,
            ) -> FakeProcess:
                recorded_commands.append(command)
                self.assertEqual(command[command.index("--negative-prompt") + 1], expected_negative_prompt)
                self.assertFalse(kwargs["shell"])
                prompt_path = Path(command[command.index("--prompt-file") + 1])
                self.assertEqual(prompt_path.read_text(encoding="utf-8"), "front view cupboard")
                Path(command[command.index("--output") + 1]).write_bytes(_png_bytes())
                return FakeProcess()

            with self.subTest(negative_prompt=negative_prompt), patch.object(
                backend.subprocess, "Popen", side_effect=start
            ):
                backend.generate_stable_diffusion_reference(
                    _png_bytes(), prompt="front view cupboard", negative_prompt=negative_prompt
                )
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0].count("--negative-prompt"), 1)

    def test_invalid_negative_prompts_do_not_download_or_start_processes(self) -> None:
        invalid_prompts = (None, 7, True, ["mask edge"], "x" * 16_001, "mask\x00edge")
        with patch.object(backend.subprocess, "Popen") as start:
            for negative_prompt in invalid_prompts:
                with self.subTest(negative_prompt_type=type(negative_prompt).__name__), self.assertRaisesRegex(
                    backend.StableDiffusionReferenceError, "negative prompt"
                ):
                    backend.generate_stable_diffusion_reference(
                        _png_bytes(), prompt="front view cupboard", negative_prompt=negative_prompt
                    )
        self.ensure_model.assert_not_called()
        start.assert_not_called()
        self.assertFalse(self.lock.locked())

    def test_cancelled_process_is_terminated_and_gpu_lock_released(self) -> None:
        for require_kill in (False, True):
            with self.subTest(require_kill=require_kill):
                state = {"cancelled": False}
                process = HangingProcess(state, require_kill=require_kill)
                with (
                    patch.object(backend.subprocess, "Popen", return_value=process),
                    self.assertRaises(backend.StableDiffusionReferenceCancelled),
                ):
                    backend.generate_stable_diffusion_reference(
                        _png_bytes(), prompt="clean table", cancellation_check=lambda state=state: state["cancelled"]
                    )
                self.assertTrue(process.terminated)
                self.assertEqual(process.killed, require_kill)
                self.assertFalse(self.lock.locked())

    def test_cancelled_while_waiting_does_not_take_another_callers_gpu_lock(self) -> None:
        self.lock.acquire()
        with self.assertRaises(backend.StableDiffusionReferenceCancelled):
            backend._acquire_cancellable(self.lock, lambda: True)
        self.assertTrue(self.lock.locked())
        self.lock.release()

    def test_cancellation_before_generation_does_not_download_or_start_process(self) -> None:
        with (
            patch.object(backend.subprocess, "Popen") as start,
            self.assertRaises(backend.StableDiffusionReferenceCancelled),
        ):
            backend.generate_stable_diffusion_reference(
                _png_bytes(), prompt="clean table", cancellation_check=lambda: True
            )
        start.assert_not_called()
        self.ensure_model.assert_not_called()
        self.assertFalse(self.lock.locked())

    def test_missing_runtime_does_not_download_checkpoint(self) -> None:
        self.executable.unlink()
        with self.assertRaisesRegex(backend.StableDiffusionReferenceError, "runtime is missing"):
            backend.generate_stable_diffusion_reference(_png_bytes(), prompt="clean table")
        self.ensure_model.assert_not_called()

    def test_invalid_inputs_do_not_start_downloads_or_processes(self) -> None:
        invalid_cases = (
            {"prompt": ""}, {"prompt": "x" * 16_001}, {"strength": 0},
            {"strength": float("nan")}, {"strength": True}, {"strength": "0.4"},
            {"output_size": (1023, 1024)}, {"output_size": (2048, 2048)},
            {"output_size": (True, 1024)}, {"image_bytes": b"invalid"},
            {"image_bytes": _png_bytes((512, 512))},
        )
        for overrides in invalid_cases:
            values = {"image_bytes": _png_bytes(), "prompt": "clean table", **overrides}
            with self.subTest(overrides=list(overrides)), self.assertRaises(backend.StableDiffusionReferenceError):
                backend.generate_stable_diffusion_reference(**values)
        self.ensure_model.assert_not_called()

    def test_start_failure_releases_lock(self) -> None:
        with (
            patch.object(backend.subprocess, "Popen", side_effect=OSError("fixture failure")),
            self.assertRaisesRegex(backend.StableDiffusionReferenceError, "could not start"),
        ):
            backend.generate_stable_diffusion_reference(_png_bytes(), prompt="clean table")
        self.assertFalse(self.lock.locked())

    def test_reports_memory_error_without_echoing_prompt_or_paths(self) -> None:
        secret = "private object prompt C:\\private\\object.png"

        def start(_command: list[str], **kwargs: object) -> FakeProcess:
            kwargs["stdout"].write(f"{secret}\nCUDA out of memory\n".encode())
            kwargs["stdout"].flush()
            return FakeProcess(7)

        with (
            patch.object(backend.subprocess, "Popen", side_effect=start),
            self.assertRaisesRegex(backend.StableDiffusionReferenceError, "insufficient GPU") as caught,
        ):
            backend.generate_stable_diffusion_reference(_png_bytes(), prompt=secret)
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("exit 7", str(caught.exception))
        self.assertFalse(self.lock.locked())

    def test_rejects_wrong_sized_or_missing_output(self) -> None:
        for output in (None, _png_bytes((512, 512))):
            def start(command: list[str], output: bytes | None = output, **_kwargs: object) -> FakeProcess:
                if output is not None:
                    Path(command[command.index("--output") + 1]).write_bytes(output)
                return FakeProcess()

            with (
                self.subTest(output_present=output is not None),
                patch.object(backend.subprocess, "Popen", side_effect=start),
                self.assertRaises(backend.StableDiffusionReferenceError),
            ):
                backend.generate_stable_diffusion_reference(_png_bytes(), prompt="clean table")
            self.assertFalse(self.lock.locked())


# ### Checkpoint download tests ###
class StableDiffusionCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.payload = _checkpoint_bytes()
        self.path = self.directory / backend.MODEL_FILENAME
        self.patches = (
            patch.object(backend, "default_stable_diffusion_model_directory", return_value=self.directory),
            patch.object(backend, "MODEL_SIZE_BYTES", len(self.payload)),
            patch.object(backend, "MODEL_SHA256", hashlib.sha256(self.payload).hexdigest()),
            patch.object(backend, "DOWNLOAD_CHUNK_BYTES", 7),
        )
        for item in self.patches:
            item.start()

    def tearDown(self) -> None:
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def test_download_is_checksummed_atomic_and_reused(self) -> None:
        progress = []
        with patch.object(backend, "urlopen", return_value=BytesIO(self.payload)) as opener:
            result = backend.ensure_stable_diffusion_model(progress_callback=lambda *args: progress.append(args))
            cached = backend.ensure_stable_diffusion_model()
        self.assertEqual(result, self.path)
        self.assertEqual(cached, self.path)
        self.assertEqual(self.path.read_bytes(), self.payload)
        self.assertEqual(opener.call_count, 1)
        self.assertEqual(progress[-1][1], 100)
        self.assertEqual(list(self.directory.glob("*.part")), [])

    def test_checksum_failure_never_installs_model(self) -> None:
        with (
            patch.object(backend, "urlopen", return_value=BytesIO(b"bad" + self.payload[3:])),
            self.assertRaisesRegex(backend.StableDiffusionReferenceError, "checksum"),
        ):
            backend.ensure_stable_diffusion_model()
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_incomplete_and_oversize_downloads_clean_up(self) -> None:
        for payload in (self.payload[:-1], self.payload + b"extra"):
            with (
                self.subTest(size=len(payload)),
                patch.object(backend, "urlopen", return_value=BytesIO(payload)),
                self.assertRaises(backend.StableDiffusionReferenceError),
            ):
                backend.ensure_stable_diffusion_model()
            self.assertEqual(list(self.directory.iterdir()), [])

    def test_cancellation_removes_partial_download_and_releases_download_lock(self) -> None:
        state = {"cancelled": False}

        def progress(_message: str, percent: int) -> None:
            if percent > 0:
                state["cancelled"] = True

        with (
            patch.object(backend, "urlopen", return_value=BytesIO(self.payload)),
            self.assertRaises(backend.StableDiffusionReferenceCancelled),
        ):
            backend.ensure_stable_diffusion_model(
                cancellation_check=lambda: state["cancelled"], progress_callback=progress
            )
        self.assertFalse(backend._MODEL_DOWNLOAD_LOCK.locked())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_network_error_is_exposed_without_installing_a_model(self) -> None:
        error = HTTPError(backend.MODEL_DOWNLOAD_URL, 403, "Forbidden", None, None)
        with (
            patch.object(backend, "urlopen", side_effect=error),
            self.assertRaisesRegex(backend.StableDiffusionReferenceError, "HTTP 403"),
        ):
            backend.ensure_stable_diffusion_model()
        self.assertFalse(self.path.exists())

    def test_insufficient_disk_space_is_detected_before_network_access(self) -> None:
        with (
            patch.object(backend.shutil, "disk_usage", return_value=SimpleNamespace(free=1)),
            patch.object(backend, "urlopen") as opener,
            self.assertRaisesRegex(backend.StableDiffusionReferenceError, "free disk space"),
        ):
            backend.ensure_stable_diffusion_model()
        opener.assert_not_called()

    def test_invalid_cached_checkpoint_is_replaced_only_after_valid_download(self) -> None:
        self.path.write_bytes(b"invalid old checkpoint")
        with patch.object(backend, "urlopen", return_value=BytesIO(self.payload)):
            backend.ensure_stable_diffusion_model()
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_failed_cache_repair_preserves_previous_file_and_cleans_up(self) -> None:
        previous = b"invalid old checkpoint"
        self.path.write_bytes(previous)
        with (
            patch.object(backend, "urlopen", return_value=BytesIO(self.payload[:-1])),
            self.assertRaises(backend.StableDiffusionReferenceError),
        ):
            backend.ensure_stable_diffusion_model()
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_checkpoint_with_wrong_header_is_redownloaded_even_if_size_matches(self) -> None:
        self.path.write_bytes(bytes(len(self.payload)))
        with patch.object(backend, "urlopen", return_value=BytesIO(self.payload)) as opener:
            backend.ensure_stable_diffusion_model()
        opener.assert_called_once()
        self.assertEqual(self.path.read_bytes(), self.payload)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
