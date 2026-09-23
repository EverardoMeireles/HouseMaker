# ### Imports ###
from __future__ import annotations

import subprocess
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from housemaker.qwen_plan_image_correction import (
    QWEN_CFG_SCALE,
    QWEN_SAMPLING_METHOD,
    QwenPlanCorrectionCancelled,
    QwenPlanCorrectionError,
    QwenSdCppModelBundle,
    QwenSdCppPlanImageEditor,
    default_qwen_model_bundle,
)


# ### Test doubles ###
class CompletedProcess:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.terminated = False
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class HangingProcess(CompletedProcess):
    def __init__(self, cancellation_state: dict[str, bool]) -> None:
        super().__init__()
        self.cancellation_state = cancellation_state

    def wait(self, timeout: float | None = None) -> int:
        if self.terminated or self.killed:
            return -1
        self.cancellation_state["cancelled"] = True
        raise subprocess.TimeoutExpired("sd-cli", timeout)


# ### Fixture helpers ###
def _png_bytes(size: tuple[int, int], color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def _write_file(path: Path, payload: bytes = b"fixture") -> Path:
    path.write_bytes(payload)
    return path


def _create_bundle(directory: Path) -> QwenSdCppModelBundle:
    return QwenSdCppModelBundle(
        executable=_write_file(directory / "sd-cli.exe"),
        diffusion_model=_write_file(directory / "qwen_image_2.1-Q6_K.gguf"),
        llm_model=_write_file(
            directory / "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
        ),
        vision_model=_write_file(
            directory / "mmproj-Qwen3VL-8B-Instruct-F16.gguf"
        ),
        vae_model=_write_file(
            directory / "qwen_image_2.1_vae_bf16.safetensors"
        ),
    )


# ### Backend tests ###
class QwenSdCppPlanImageEditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.bundle = _create_bundle(self.directory)
        self.editor = QwenSdCppPlanImageEditor(self.bundle)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_runs_official_qwen_edit_command_and_returns_exact_png(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []

        def start_process(command: list[str], **kwargs: object) -> CompletedProcess:
            calls.append((command, kwargs))
            output_path = Path(command[command.index("-o") + 1])
            Image.new("RGB", (1024, 768), "white").save(output_path, "PNG")
            return CompletedProcess()

        with patch(
            "housemaker.qwen_plan_image_correction.subprocess.Popen",
            side_effect=start_process,
        ):
            result = self.editor(
                _png_bytes((1024, 768)),
                prompt="Correct this architectural plan.",
                output_size=(1024, 768),
                cancellation_check=None,
            )

        with Image.open(BytesIO(result)) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (1024, 768))
        command, kwargs = calls[0]
        expected_pairs = {
            "--diffusion-model": self.bundle.diffusion_model,
            "--vae": self.bundle.vae_model,
            "--llm": self.bundle.llm_model,
            "--llm_vision": self.bundle.vision_model,
            "--cfg-scale": QWEN_CFG_SCALE,
            "--sampling-method": QWEN_SAMPLING_METHOD,
            "--steps": "40",
            "-W": "1024",
            "-H": "768",
        }
        for option, expected in expected_pairs.items():
            self.assertIn(option, command)
            normalized = str(expected.resolve()) if isinstance(expected, Path) else expected
            self.assertEqual(command[command.index(option) + 1], normalized)
        self.assertIn("-r", command)
        self.assertIn("-p", command)
        self.assertIn("--offload-to-cpu", command)
        self.assertIn("--diffusion-fa", command)
        self.assertIn("--vae-tiling", command)
        self.assertEqual(command[command.index("-p") + 1], "Correct this architectural plan.")
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)

    def test_cancellation_terminates_running_process(self) -> None:
        state = {"cancelled": False}
        process = HangingProcess(state)
        with (
            patch(
                "housemaker.qwen_plan_image_correction.subprocess.Popen",
                return_value=process,
            ),
            self.assertRaises(QwenPlanCorrectionCancelled),
        ):
            self.editor(
                _png_bytes((1024, 1024)),
                prompt="Correct plan.",
                output_size=(1024, 1024),
                cancellation_check=lambda: state["cancelled"],
            )
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)

    def test_process_diagnostics_are_not_exposed(self) -> None:
        secret = "private-path-or-driver-detail"

        def fail_process(command: list[str], **kwargs: object) -> CompletedProcess:
            log_file = kwargs["stdout"]
            log_file.write(secret.encode())
            log_file.flush()
            return CompletedProcess(return_code=3)

        with (
            patch(
                "housemaker.qwen_plan_image_correction.subprocess.Popen",
                side_effect=fail_process,
            ),
            self.assertRaises(QwenPlanCorrectionError) as caught,
        ):
            self.editor(
                _png_bytes((1024, 1024)),
                prompt="Correct plan.",
                output_size=(1024, 1024),
                cancellation_check=None,
            )
        self.assertNotIn(secret, str(caught.exception))

    def test_rejects_missing_or_wrong_size_process_output(self) -> None:
        def wrong_size_process(command: list[str], **_kwargs: object) -> CompletedProcess:
            output_path = Path(command[command.index("-o") + 1])
            Image.new("RGB", (512, 512), "white").save(output_path, "PNG")
            return CompletedProcess()

        with (
            patch(
                "housemaker.qwen_plan_image_correction.subprocess.Popen",
                side_effect=wrong_size_process,
            ),
            self.assertRaisesRegex(QwenPlanCorrectionError, "unexpected image size"),
        ):
            self.editor(
                _png_bytes((1024, 1024)),
                prompt="Correct plan.",
                output_size=(1024, 1024),
                cancellation_check=None,
            )

    def test_validates_prepared_png_and_multiple_of_32_dimensions(self) -> None:
        invalid_cases = (
            (b"not-png", (1024, 1024)),
            (_png_bytes((1024, 1024)), (1000, 1024)),
            (_png_bytes((1024, 1024)), (1024, 768)),
        )
        for payload, size in invalid_cases:
            with self.subTest(size=size), self.assertRaises(QwenPlanCorrectionError):
                self.editor(
                    payload,
                    prompt="Correct plan.",
                    output_size=size,
                    cancellation_check=None,
                )


# ### Bundle tests ###
class QwenSdCppBundleTests(unittest.TestCase):
    def test_default_bundle_uses_local_app_data(self) -> None:
        with patch.dict(
            "os.environ",
            {"LOCALAPPDATA": str(Path("C:/LocalAppData"))},
        ):
            bundle = default_qwen_model_bundle()
        expected = Path("C:/LocalAppData/HouseMaker/models/qwen-image-2.1")
        self.assertEqual(bundle.executable, expected / "runtime/sd-cli.exe")
        self.assertEqual(
            bundle.diffusion_model,
            expected / "weights/qwen_image_2.1-Q6_K.gguf",
        )

    def test_rejects_incompatible_quantization_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            bundle = _create_bundle(directory)
            wrong_diffusion = _write_file(directory / "qwen_image_2.1-Q4_K.gguf")
            invalid = QwenSdCppModelBundle(
                bundle.executable,
                wrong_diffusion,
                bundle.llm_model,
                bundle.vision_model,
                bundle.vae_model,
            )
            with self.assertRaisesRegex(
                QwenPlanCorrectionError,
                "unsupported format or quantization",
            ):
                QwenSdCppPlanImageEditor(invalid)

    def test_accepts_documented_bf16_mmproj_filename(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            bundle = _create_bundle(directory)
            bf16 = _write_file(
                directory / "Qwen3VL-8B-Instruct-mmproj-BF16.gguf"
            )
            configured = QwenSdCppModelBundle(
                bundle.executable,
                bundle.diffusion_model,
                bundle.llm_model,
                bf16,
                bundle.vae_model,
            )
            editor = QwenSdCppPlanImageEditor(configured)
            self.assertIsInstance(editor, QwenSdCppPlanImageEditor)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
