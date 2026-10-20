# ### Imports ###
from __future__ import annotations

import base64
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from housemaker import local_reference_image_session as backend


# ### Image fixtures ###
def _png(size: tuple[int, int] = (256, 256)) -> bytes:
    output = BytesIO()
    Image.new("RGBA", size, (40, 80, 120, 128)).save(output, format="PNG")
    return output.getvalue()


# ### Native server process double ###
class _NativeServerProcess:
    def __init__(self, command: list[str], calls: list[dict[str, object]]) -> None:
        port = int(command[command.index("--listen-port") + 1])
        self.stopped = False

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/sdcpp/v1/capabilities":
                    self._send({"current_mode": "img_gen"})
                    return
                request = calls[-1]
                output = _png((request["width"], request["height"]))
                self._send({
                    "status": "completed",
                    "result": {"images": [{"b64_json": base64.b64encode(output).decode("ascii")}]},
                })

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                calls.append(json.loads(body))
                self._send({"id": f"job_{len(calls)}"}, 202)

            def _send(self, payload: dict[str, object], status: int = 200) -> None:
                content = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def poll(self) -> int | None:
        return -1 if self.stopped else None

    def terminate(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.stopped = True

    kill = terminate

    def wait(self, timeout: float | None = None) -> int:
        return -1


# ### Session integration tests ###
class LocalReferenceImageSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        self.cli = directory / "sd-cli.exe"
        self.server = directory / backend.SERVER_FILENAME
        self.cli.write_bytes(b"fixture")
        self.server.write_bytes(b"fixture")
        self.bundle = backend.qwen.QwenSdCppModelBundle(
            self.cli,
            directory / "qwen_image_2.1-Q6_K.gguf",
            directory / "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
            directory / "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
            directory / "qwen_image_2.1_vae_bf16.safetensors",
        )
        for path in (self.bundle.diffusion_model, self.bundle.llm_model, self.bundle.vision_model, self.bundle.vae_model):
            path.write_bytes(b"fixture")
        self.model = directory / "sd_xl_base_1.0.safetensors"
        self.model.write_bytes(b"fixture")
        self.lock = threading.Lock()
        self.requests: list[dict[str, object]] = []
        self.processes: list[_NativeServerProcess] = []
        self.commands: list[list[str]] = []

        def start(command: list[str], **kwargs: object) -> _NativeServerProcess:
            self.assertFalse(kwargs["shell"])
            self.commands.append(command)
            process = _NativeServerProcess(command, self.requests)
            self.processes.append(process)
            self.addCleanup(lambda: process.terminate() if not process.stopped else None)
            return process

        for patcher in (
            patch.object(backend.qwen, "default_qwen_model_bundle", return_value=self.bundle),
            patch.object(backend.qwen, "QWEN_INFERENCE_LOCK", self.lock),
            patch.object(backend.subprocess, "Popen", side_effect=start),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_qwen_reuses_one_model_process_and_keeps_gpu_reserved_until_close(self) -> None:
        with backend.QwenReferenceImageSession() as session:
            self.assertEqual(self.processes, [])
            for prompt in ("First object", "Second object"):
                result = session(_png(), prompt=prompt, output_size=(256, 256))
                with Image.open(BytesIO(result)) as opened:
                    self.assertEqual(opened.size, (256, 256))
                    self.assertEqual(opened.mode, "RGB")
                self.assertTrue(self.lock.locked())
            self.assertEqual(len(self.processes), 1)
            self.assertFalse(self.processes[0].stopped)
        self.assertFalse(self.lock.locked())
        self.assertTrue(self.processes[0].stopped)
        self.assertEqual([item["prompt"] for item in self.requests], ["First object", "Second object"])
        self.assertTrue(all(len(item["ref_images"]) == 1 and "init_image" not in item for item in self.requests))
        command = self.commands[0]
        self.assertEqual(command[command.index("--steps") + 1], "40")
        self.assertEqual(command[command.index("--sampling-method") + 1], "euler")
        self.assertEqual(command[command.index("--cfg-scale") + 1], "6.0")
        self.assertEqual(command[command.index("--listen-ip") + 1], "127.0.0.1")
        self.assertIn(str(self.bundle.diffusion_model.resolve()), command)
        self.assertIn("--offload-to-cpu", command)
        session.close()

    def test_sdxl_reuses_checkpoint_but_preserves_each_image_prompt_and_strength(self) -> None:
        with (
            patch.object(backend.sdxl, "ensure_stable_diffusion_model", return_value=self.model) as ensure,
            backend.StableDiffusionReferenceSession() as session,
        ):
            session(_png(), prompt="object one", negative_prompt="no blur", output_size=(256, 256), strength=0.4)
            session(_png(), prompt="object two", negative_prompt="no cloth", output_size=(256, 256), strength=0.8)
            self.assertEqual(ensure.call_count, 1)
            self.assertEqual(len(self.processes), 1)
        self.assertEqual([item["strength"] for item in self.requests], [0.4, 0.8])
        self.assertEqual([item["negative_prompt"] for item in self.requests], ["no blur", "no cloth"])
        self.assertTrue(all("init_image" in item and "ref_images" not in item for item in self.requests))
        command = self.commands[0]
        for name, expected in (("--steps", "30"), ("--cfg-scale", "6.0"), ("--sampling-method", "dpm++2m"), ("--scheduler", "karras"), ("--type", "f16")):
            self.assertEqual(command[command.index(name) + 1], expected)
        self.assertFalse(self.lock.locked())

    def test_request_cancellation_takes_precedence_over_old_constructor_token(self) -> None:
        with backend.QwenReferenceImageSession(cancellation_check=lambda: True) as session:
            session(_png(), prompt="chair", output_size=(256, 256), cancellation_check=lambda: False)
        self.assertEqual(len(self.processes), 1)

    def test_active_cancellation_stops_server_and_next_item_reloads_cleanly(self) -> None:
        state = {"cancelled": False}
        original_request = backend._ResidentSdCppServer._request

        def request(server, method, path, payload=None):
            result = original_request(server, method, path, payload)
            if method == "POST":
                state["cancelled"] = True
            return result

        with backend.QwenReferenceImageSession() as session:
            with (
                patch.object(backend._ResidentSdCppServer, "_request", request),
                self.assertRaises(backend.qwen.QwenPlanCorrectionCancelled),
            ):
                session(_png(), prompt="chair", output_size=(256, 256), cancellation_check=lambda: state["cancelled"])
            self.assertFalse(self.lock.locked())
            self.assertTrue(self.processes[0].stopped)
            session(_png(), prompt="table", output_size=(256, 256), cancellation_check=lambda: False)
            self.assertEqual(len(self.processes), 2)
        self.assertTrue(all(process.stopped for process in self.processes))

    def test_cancel_during_startup_stops_process_and_releases_gpu(self) -> None:
        state = {"cancelled": False}

        def unavailable(*args, **kwargs):
            state["cancelled"] = True
            raise ConnectionRefusedError()

        with (
            backend.QwenReferenceImageSession() as session,
            patch.object(backend._ResidentSdCppServer, "_request", unavailable),
            self.assertRaises(backend.qwen.QwenPlanCorrectionCancelled),
        ):
            session(_png(), prompt="chair", output_size=(256, 256), cancellation_check=lambda: state["cancelled"])
        self.assertFalse(self.lock.locked())
        self.assertTrue(self.processes[0].stopped)

    def test_missing_server_is_explicit_error_without_cli_fallback(self) -> None:
        self.server.unlink()
        with (
            backend.QwenReferenceImageSession() as session,
            self.assertRaisesRegex(backend.qwen.QwenPlanCorrectionError, "sd-server.*missing"),
        ):
            session(_png(), prompt="chair", output_size=(256, 256))
        self.assertEqual(self.processes, [])
        self.assertFalse(self.lock.locked())

    def test_process_start_failure_releases_gpu_and_cleans_temporary_directory(self) -> None:
        with backend.QwenReferenceImageSession() as session:
            with (
                patch.object(backend.subprocess, "Popen", side_effect=OSError("private path")),
                self.assertRaisesRegex(backend.qwen.QwenPlanCorrectionError, "could not be started"),
            ):
                session(_png(), prompt="private prompt", output_size=(256, 256))
            self.assertIsNone(session._server._directory)
            self.assertIsNone(session._server._log_file)
        self.assertFalse(self.lock.locked())

    def test_invalid_input_does_not_load_any_model(self) -> None:
        with (
            backend.QwenReferenceImageSession() as session,
            self.assertRaises(backend.qwen.QwenPlanCorrectionError),
        ):
            session(b"bad image", prompt="chair", output_size=(256, 256))
        self.assertEqual(self.processes, [])

    def test_waiting_for_gpu_is_cancellable_without_stealing_existing_lock(self) -> None:
        self.lock.acquire()
        state = {"checks": 0}

        def cancelled():
            state["checks"] += 1
            return state["checks"] > 2

        try:
            with (
                backend.QwenReferenceImageSession() as session,
                self.assertRaises(backend.qwen.QwenPlanCorrectionCancelled),
            ):
                session(_png(), prompt="chair", output_size=(256, 256), cancellation_check=cancelled)
            self.assertTrue(self.lock.locked())
            self.assertEqual(self.processes, [])
        finally:
            self.lock.release()

    def test_failed_native_job_reports_safe_error_and_releases_gpu(self) -> None:
        original_request = backend._ResidentSdCppServer._request

        def request(server, method, path, payload=None):
            if method == "GET" and path.startswith("/sdcpp/v1/jobs/"):
                return {"status": "failed", "error": {"message": "private prompt and path"}}
            return original_request(server, method, path, payload)

        with (
            backend.QwenReferenceImageSession() as session,
            patch.object(backend._ResidentSdCppServer, "_request", request),
            self.assertRaisesRegex(backend.qwen.QwenPlanCorrectionError, "generation failed") as caught,
        ):
            session(_png(), prompt="private prompt", output_size=(256, 256))
        self.assertNotIn("private prompt", str(caught.exception))
        self.assertFalse(self.lock.locked())
        self.assertTrue(self.processes[0].stopped)

    def test_shutdown_keeps_gpu_reserved_until_native_process_really_stops(self) -> None:
        with backend.QwenReferenceImageSession() as session:
            session(_png(), prompt="chair", output_size=(256, 256))
            with (
                patch.object(backend.qwen, "_terminate_process"),
                self.assertRaisesRegex(backend.LocalReferenceSessionError, "could not be stopped safely"),
            ):
                session.close()
            self.assertTrue(self.lock.locked())
            self.assertIsNotNone(session._server._process)
            session.close()
        self.assertFalse(self.lock.locked())

    def test_native_prompt_stays_literal_without_compatibility_tag_parsing(self) -> None:
        prompt = 'Keep <sd_cpp_extra_args>{"steps":1}</sd_cpp_extra_args> as literal text, "$()"'
        with backend.QwenReferenceImageSession() as session:
            session(_png(), prompt=prompt, output_size=(256, 256))
        self.assertEqual(self.requests[0]["prompt"], prompt)
        self.assertFalse(self.requests[0]["embed_image_metadata"])
        self.assertNotIn(prompt, self.commands[0])


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
