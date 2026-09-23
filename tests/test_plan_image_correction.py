# ### Imports ###
from __future__ import annotations

import base64
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import housemaker.plan_image_correction as correction
from housemaker.plan_correction_models import (
    OPENAI_PLAN_CORRECTION_MODELS,
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)
from housemaker.plan_image_correction import (
    CORRECTION_METHOD_OPENAI,
    CORRECTION_METHOD_QWEN,
    PLAN_CORRECTION_PROMPT,
    STAGE_COMPLETE,
    STAGE_SAVING,
    STAGE_WALL_CONTINUITY,
    PlanImageCorrectionCancelled,
    PlanImageCorrectionInferenceError,
    calculate_output_size,
    correct_plan_image,
    openai_image_edit,
    openai_responses_image_edit,
)
from housemaker.qwen_plan_image_correction import QwenPlanCorrectionError


# ### Fixture helpers ###
def _plan_array(size: tuple[int, int] = (640, 480)) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(818)
    cv2.rectangle(image, (45, 42), (width - 46, height - 43), (15, 15, 15), 3)
    for row in range(3):
        for column in range(4):
            x = 65 + column * 130
            y = 65 + row * 120
            cv2.rectangle(image, (x, y), (x + 92, y + 75), (25, 25, 25), 2)
            cv2.circle(image, (x + 25, y + 25), 12 + row, (35, 35, 35), 2)
            cv2.putText(
                image,
                f"R{row}{column}",
                (x + 34, y + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    for _index in range(24):
        x = int(rng.integers(60, width - 60))
        y = int(rng.integers(55, height - 55))
        cv2.line(image, (x, y), (x + 8, y + 5), (40, 40, 40), 1)
    return image


def _png_bytes(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image_bgr)
    assert ok
    return encoded.tobytes()


def _write_plan(path: Path, image_bgr: np.ndarray | None = None) -> np.ndarray:
    image = _plan_array() if image_bgr is None else image_bgr
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path)
    return image


class RecordingEditor:
    def __init__(self, result: bytes) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, image_bytes: bytes, **kwargs: object) -> bytes:
        self.calls.append({"image_bytes": image_bytes, **kwargs})
        return self.result


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.position = 0

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self.payload) - self.position
        start = self.position
        self.position = min(len(self.payload), start + amount)
        return self.payload[start : self.position]

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None


# ### Pipeline tests ###
class PlanImageCorrectionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.input_path = self.directory / "photo.png"
        self.output_path = self.directory / "corrected.png"
        self.plan = _write_plan(self.input_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_ai_edit_uses_exact_prompt_and_writes_strict_binary_png(self) -> None:
        expected_size = calculate_output_size(640, 480)
        candidate = correction._fit_complete_image(self.plan, expected_size)
        editor = RecordingEditor(_png_bytes(candidate))
        progress = []
        result = correct_plan_image(
            self.input_path,
            self.output_path,
            api_key="test-key",
            image_editor=editor,
            progress_callback=progress.append,
        )
        self.assertEqual(result.method, CORRECTION_METHOD_OPENAI)
        self.assertEqual(result.input_size, (640, 480))
        self.assertEqual(result.output_size, expected_size)
        self.assertEqual(editor.calls[0]["prompt"], PLAN_CORRECTION_PROMPT)
        self.assertEqual(editor.calls[0]["output_size"], expected_size)
        prepared_input = cv2.imdecode(
            np.frombuffer(editor.calls[0]["image_bytes"], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(prepared_input)
        assert prepared_input is not None
        margin = round(
            min(expected_size) * correction.EDITOR_SAFE_MARGIN_FRACTION
        )
        self.assertTrue(np.all(prepared_input[:margin] == 255))
        self.assertTrue(np.all(prepared_input[-margin:] == 255))
        self.assertTrue(np.all(prepared_input[:, :margin] == 255))
        self.assertTrue(np.all(prepared_input[:, -margin:] == 255))
        with Image.open(self.output_path) as image:
            pixels = np.array(image.convert("L"))
        self.assertEqual(set(np.unique(pixels)), {0, 255})
        self.assertEqual(progress[-1].stage, STAGE_COMPLETE)
        self.assertEqual(progress[-1].percent, 100)

    def test_wall_continuity_postprocessing_is_opt_in(self) -> None:
        expected_size = calculate_output_size(640, 480)
        candidate = correction._fit_complete_image(self.plan, expected_size)
        with patch.object(
            correction,
            "make_doorway_walls_continuous",
        ) as processor:
            correct_plan_image(
                self.input_path,
                self.output_path,
                api_key="test-key",
                image_editor=RecordingEditor(_png_bytes(candidate)),
            )

        processor.assert_not_called()

    def test_wall_continuity_runs_after_binarization_when_enabled(self) -> None:
        expected_size = calculate_output_size(640, 480)
        candidate = correction._fit_complete_image(self.plan, expected_size)

        def mark_processed(binary: np.ndarray) -> np.ndarray:
            self.assertEqual(binary.dtype, np.uint8)
            self.assertEqual(set(np.unique(binary)), {0, 255})
            processed = binary.copy()
            processed[0, 0] = 0
            return processed

        with patch.object(
            correction,
            "make_doorway_walls_continuous",
            side_effect=mark_processed,
        ) as processor:
            result = correct_plan_image(
                self.input_path,
                self.output_path,
                api_key="test-key",
                image_editor=RecordingEditor(_png_bytes(candidate)),
                make_walls_continuous=True,
            )

        processor.assert_called_once()
        self.assertIn(STAGE_WALL_CONTINUITY, result.stages)
        with Image.open(self.output_path) as image:
            pixels = np.array(image.convert("L"))
        self.assertEqual(int(pixels[0, 0]), 0)

    def test_prompt_does_not_request_continuous_doorway_walls(self) -> None:
        normalized = PLAN_CORRECTION_PROMPT.casefold()
        self.assertNotIn("one continuous wall", normalized)
        self.assertNotIn("extend both interrupted", normalized)
        self.assertNotIn("opposite jamb", normalized)
        self.assertNotIn("sole permitted architectural line addition", normalized)

    def test_prompt_removes_measurement_lines_and_values(self) -> None:
        normalized = " ".join(PLAN_CORRECTION_PROMPT.casefold().split())
        self.assertIn("remove every measurement annotation", normalized)
        self.assertIn("dimension and measurement lines", normalized)
        self.assertIn("all numeric values or unit text", normalized)
        self.assertIn("non-measurement room or feature label", normalized)
        self.assertNotIn("labels, dimensions", normalized)
        self.assertNotIn("changed dimensions", normalized)

    def test_missing_key_fails_without_writing_an_output(self) -> None:
        for model in OPENAI_PLAN_CORRECTION_MODELS:
            with self.subTest(model=model), self.assertRaisesRegex(
                PlanImageCorrectionInferenceError,
                "not configured",
            ):
                correct_plan_image(
                    self.input_path,
                    self.output_path,
                    model=model,
                )
            self.assertFalse(self.output_path.exists())

    def test_gpt_5_6_models_use_responses_editor_and_stay_openai(self) -> None:
        expected_size = calculate_output_size(640, 480)
        candidate = correction._fit_complete_image(self.plan, expected_size)
        for model in (
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        ):
            with self.subTest(model=model):
                output_path = self.directory / f"{model}.png"
                with patch.object(
                    correction,
                    "openai_responses_image_edit",
                    return_value=_png_bytes(candidate),
                ) as edit:
                    result = correct_plan_image(
                        self.input_path,
                        output_path,
                        api_key="test-key",
                        model=model,
                    )
                self.assertEqual(result.method, CORRECTION_METHOD_OPENAI)
                self.assertTrue(output_path.is_file())
                self.assertEqual(edit.call_args.kwargs["model"], model)
                self.assertEqual(
                    edit.call_args.kwargs["prompt"],
                    PLAN_CORRECTION_PROMPT,
                )

    def test_qwen_editor_uses_shared_pipeline_without_openai_key(self) -> None:
        expected_size = calculate_output_size(640, 480)
        candidate = correction._fit_complete_image(self.plan, expected_size)
        editor = RecordingEditor(_png_bytes(candidate))
        result = correct_plan_image(
            self.input_path,
            self.output_path,
            model=PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            image_editor=editor,
        )
        self.assertEqual(result.method, CORRECTION_METHOD_QWEN)
        self.assertEqual(editor.calls[0]["api_key"], "")
        self.assertEqual(editor.calls[0]["output_size"], expected_size)

    def test_editor_border_is_preserved(self) -> None:
        expected_size = calculate_output_size(640, 480)
        candidate = correction._fit_complete_image(self.plan, expected_size)
        candidate[0] = 0
        candidate[-1] = 0
        candidate[:, 0] = 0
        candidate[:, -1] = 0
        result = correct_plan_image(
            self.input_path,
            self.output_path,
            api_key="test-key",
            image_editor=RecordingEditor(_png_bytes(candidate)),
        )
        self.assertEqual(result.method, CORRECTION_METHOD_OPENAI)
        with Image.open(self.output_path) as image:
            pixels = np.array(image.convert("L"))
        self.assertTrue(np.all(pixels[0] == 0))
        self.assertTrue(np.all(pixels[-1] == 0))
        self.assertTrue(np.all(pixels[:, 0] == 0))
        self.assertTrue(np.all(pixels[:, -1] == 0))

    def test_real_qwen_factory_dispatch_stays_on_the_worker_thread(self) -> None:
        expected_size = calculate_output_size(640, 480)
        candidate = correction._fit_complete_image(self.plan, expected_size)
        invocation_threads: list[int] = []

        def local_editor(image_bytes: bytes, **_kwargs: object) -> bytes:
            invocation_threads.append(threading.get_ident())
            return _png_bytes(candidate)

        current_thread = threading.get_ident()
        with patch.object(
            correction,
            "create_default_qwen_plan_image_editor",
            return_value=local_editor,
        ) as create_editor:
            result = correct_plan_image(
                self.input_path,
                self.output_path,
                model=PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
                cancellation_check=lambda: False,
            )

        create_editor.assert_called_once_with()
        self.assertEqual(invocation_threads, [current_thread])
        self.assertEqual(result.method, CORRECTION_METHOD_QWEN)

    def test_qwen_factory_failure_reports_safe_reason_without_output(self) -> None:
        safe_reason = "The configured Qwen diffusion model file is unavailable."
        with (
            patch.object(
                correction,
                "create_default_qwen_plan_image_editor",
                side_effect=QwenPlanCorrectionError(safe_reason),
            ),
            self.assertRaises(PlanImageCorrectionInferenceError) as caught,
        ):
            correct_plan_image(
                self.input_path,
                self.output_path,
                model=PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            )
        self.assertEqual(str(caught.exception), safe_reason)
        self.assertFalse(self.output_path.exists())

    def test_unknown_plan_correction_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            correction.PlanImageCorrectionError,
            "Unsupported plan correction model",
        ):
            correct_plan_image(
                self.input_path,
                self.output_path,
                model="unknown-model",
            )

    def test_untrusted_editor_failure_never_exposes_api_key(self) -> None:
        secret = "sk-private-do-not-log"

        def failing_editor(*_args: object, **_kwargs: object) -> bytes:
            raise RuntimeError(f"request Authorization Bearer {secret}")

        with self.assertRaises(PlanImageCorrectionInferenceError) as caught:
            correct_plan_image(
                self.input_path,
                self.output_path,
                api_key=secret,
                image_editor=failing_editor,
            )
        self.assertNotIn(secret, str(caught.exception))
        self.assertFalse(self.output_path.exists())

    def test_structurally_changed_results_are_accepted(self) -> None:
        expected_size = calculate_output_size(640, 480)
        base = correction._fit_complete_image(self.plan, expected_size)

        cropped = cv2.resize(
            self.plan[90:-90, 110:-110],
            expected_size,
            interpolation=cv2.INTER_CUBIC,
        )
        speckled = base.copy()
        rng = np.random.default_rng(91)
        points = rng.integers(
            (10, 10),
            (expected_size[0] - 10, expected_size[1] - 10),
            size=(2500, 2),
        )
        speckled[points[:, 1], points[:, 0]] = 0
        erased = base.copy()
        erased[
            round(expected_size[1] * 0.25) : round(expected_size[1] * 0.75),
            round(expected_size[0] * 0.25) : round(expected_size[0] * 0.75),
        ] = 255
        invented = base.copy()
        cv2.rectangle(invented, (372, 272), (572, 432), (0, 0, 0), 5)

        for name, candidate in (
            ("cropped", cropped),
            ("speckled", speckled),
            ("erased", erased),
            ("invented", invented),
        ):
            with self.subTest(name=name):
                output_path = self.directory / f"{name}.png"
                result = correct_plan_image(
                    self.input_path,
                    output_path,
                    api_key="test-key",
                    image_editor=RecordingEditor(_png_bytes(candidate)),
                )
                self.assertEqual(result.output_path, output_path)
                self.assertTrue(output_path.is_file())

    def test_wrong_ai_dimensions_are_rejected_before_image_allocation(self) -> None:
        with self.assertRaisesRegex(PlanImageCorrectionInferenceError, "dimensions"):
            correct_plan_image(
                self.input_path,
                self.output_path,
                api_key="test-key",
                image_editor=RecordingEditor(_png_bytes(self.plan)),
            )
        self.assertFalse(self.output_path.exists())

    def test_unreadable_ai_result_is_rejected(self) -> None:
        with self.assertRaisesRegex(PlanImageCorrectionInferenceError, "unreadable"):
            correct_plan_image(
                self.input_path,
                self.output_path,
                api_key="test-key",
                image_editor=RecordingEditor(b"not an image"),
            )
        self.assertFalse(self.output_path.exists())

    def test_cancellation_before_replace_preserves_existing_output(self) -> None:
        previous = b"previous output"
        self.output_path.write_bytes(previous)
        cancelled = False

        def on_progress(update) -> None:
            nonlocal cancelled
            if update.stage == STAGE_SAVING:
                cancelled = True

        with self.assertRaises(PlanImageCorrectionCancelled):
            correct_plan_image(
                self.input_path,
                self.output_path,
                api_key="test-key",
                image_editor=RecordingEditor(
                    _png_bytes(
                        correction._fit_complete_image(
                            self.plan,
                            calculate_output_size(640, 480),
                        )
                    )
                ),
                progress_callback=on_progress,
                cancellation_check=lambda: cancelled,
            )
        self.assertEqual(self.output_path.read_bytes(), previous)
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_cancellation_at_wall_continuity_preserves_existing_output(
        self,
    ) -> None:
        previous = b"previous output"
        self.output_path.write_bytes(previous)
        cancelled = False

        def on_progress(update) -> None:
            nonlocal cancelled
            if update.stage == STAGE_WALL_CONTINUITY:
                cancelled = True

        expected_size = calculate_output_size(640, 480)
        candidate = correction._fit_complete_image(self.plan, expected_size)
        with patch.object(
            correction,
            "make_doorway_walls_continuous",
        ) as processor, self.assertRaises(PlanImageCorrectionCancelled):
            correct_plan_image(
                self.input_path,
                self.output_path,
                api_key="test-key",
                image_editor=RecordingEditor(_png_bytes(candidate)),
                progress_callback=on_progress,
                cancellation_check=lambda: cancelled,
                make_walls_continuous=True,
            )

        processor.assert_not_called()
        self.assertEqual(self.output_path.read_bytes(), previous)
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_source_is_never_overwritten(self) -> None:
        original = self.input_path.read_bytes()
        with self.assertRaisesRegex(
            correction.PlanImageCorrectionError,
            "separately",
        ):
            correct_plan_image(self.input_path, self.input_path)
        self.assertEqual(self.input_path.read_bytes(), original)

    def test_cancellation_does_not_wait_for_a_stalled_editor(self) -> None:
        cancel_requested = threading.Event()
        release_editor = threading.Event()

        def stalled_editor(*_args: object, **_kwargs: object) -> bytes:
            cancel_requested.set()
            release_editor.wait(2.0)
            return _png_bytes(self.plan)

        try:
            with self.assertRaises(PlanImageCorrectionCancelled):
                correct_plan_image(
                    self.input_path,
                    self.output_path,
                    api_key="test-key",
                    image_editor=stalled_editor,
                    cancellation_check=cancel_requested.is_set,
                )
        finally:
            release_editor.set()
        self.assertFalse(self.output_path.exists())


# ### Transport tests ###
class OpenAIImageEditTests(unittest.TestCase):
    def test_request_uses_gpt_image_2_high_quality_and_omits_input_fidelity(self) -> None:
        returned = _png_bytes(_plan_array((320, 240)))
        response = json.dumps(
            {"data": [{"b64_json": base64.b64encode(returned).decode()}]}
        ).encode()
        captured = []

        def fake_open(request, **kwargs):
            captured.append((request, kwargs))
            return FakeResponse(response)

        with patch.object(correction, "urlopen", side_effect=fake_open):
            result = openai_image_edit(
                b"input-png",
                api_key="sk-test",
                prompt=PLAN_CORRECTION_PROMPT,
                output_size=(1024, 768),
                cancellation_check=None,
            )
        self.assertEqual(result, returned)
        request, kwargs = captured[0]
        body = request.data
        self.assertIn(b"gpt-image-2", body)
        self.assertIn(b"high", body)
        self.assertIn(b"1024x768", body)
        self.assertIn(b'name="image[]"', body)
        self.assertIn(PLAN_CORRECTION_PROMPT.encode(), body)
        self.assertNotIn(b"input_fidelity", body)
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-test")
        self.assertEqual(kwargs["timeout"], correction.OPENAI_NETWORK_TIMEOUT_SECONDS)

    def test_responses_edit_uses_selected_model_and_forced_image_tool(self) -> None:
        returned = _png_bytes(_plan_array((320, 240)))
        response = json.dumps(
            {
                "output": [
                    {
                        "type": "image_generation_call",
                        "result": base64.b64encode(returned).decode(),
                    }
                ]
            }
        ).encode()
        captured = []

        def fake_open(request, **kwargs):
            captured.append((request, kwargs))
            return FakeResponse(response)

        for model in (
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        ):
            with self.subTest(model=model), patch.object(
                correction,
                "urlopen",
                side_effect=fake_open,
            ):
                result = openai_responses_image_edit(
                    b"input-png",
                    model=model,
                    api_key="sk-test",
                    prompt=PLAN_CORRECTION_PROMPT,
                    output_size=(1024, 768),
                    cancellation_check=None,
                )
            self.assertEqual(result, returned)
            request, kwargs = captured.pop()
            body = json.loads(request.data.decode())
            self.assertEqual(body["model"], model)
            self.assertFalse(body["store"])
            self.assertEqual(body["tool_choice"], {"type": "image_generation"})
            self.assertEqual(
                body["input"][0]["content"][0],
                {"type": "input_text", "text": PLAN_CORRECTION_PROMPT},
            )
            image_input = body["input"][0]["content"][1]
            self.assertEqual(image_input["type"], "input_image")
            self.assertEqual(image_input["detail"], "original")
            self.assertTrue(
                image_input["image_url"].startswith("data:image/png;base64,")
            )
            self.assertEqual(
                body["tools"],
                [
                    {
                        "type": "image_generation",
                        "model": "gpt-image-2",
                        "action": "edit",
                        "quality": "high",
                        "size": "1024x768",
                        "output_format": "png",
                    }
                ],
            )
            self.assertEqual(
                request.full_url,
                correction.OPENAI_RESPONSES_URL,
            )
            self.assertEqual(request.get_header("Authorization"), "Bearer sk-test")
            self.assertEqual(
                kwargs["timeout"],
                correction.OPENAI_NETWORK_TIMEOUT_SECONDS,
            )

    def test_responses_edit_rejects_a_response_without_an_image_call(self) -> None:
        response = json.dumps(
            {"output": [{"type": "message", "content": []}]}
        ).encode()
        with (
            patch.object(
                correction,
                "urlopen",
                return_value=FakeResponse(response),
            ),
            self.assertRaisesRegex(
                PlanImageCorrectionInferenceError,
                "did not return a usable image",
            ),
        ):
            openai_responses_image_edit(
                b"input-png",
                model=PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
                api_key="sk-test",
                prompt=PLAN_CORRECTION_PROMPT,
                output_size=(1024, 768),
                cancellation_check=None,
            )


# ### Local processing tests ###
class PlanCleanupTests(unittest.TestCase):
    def test_output_size_is_bounded_and_divisible_by_sixteen(self) -> None:
        sizes = (
            (4000, 2000),
            (1671, 1671),
            (1001, 713),
            (32, 18),
            (9000, 1000),
        )
        for source in sizes:
            with self.subTest(source=source):
                width, height = calculate_output_size(*source)
                self.assertLessEqual(max(width, height), 2048)
                self.assertEqual(width % 16, 0)
                self.assertEqual(height % 16, 0)
                self.assertGreaterEqual(
                    width * height,
                    correction.MIN_OUTPUT_PIXEL_COUNT,
                )
                self.assertLessEqual(
                    max(width, height) / min(width, height),
                    correction.MAX_OUTPUT_ASPECT_RATIO,
                )
                source_ratio = source[0] / source[1]
                if 1 / 3 <= source_ratio <= 3:
                    self.assertLess(
                        abs((width / height) / source_ratio - 1),
                        0.03,
                    )

        rng = np.random.default_rng(2048)
        for source_width, source_height in rng.integers(
            1,
            correction.MAX_INPUT_EDGE_PIXELS + 1,
            size=(500, 2),
        ):
            width, height = calculate_output_size(
                int(source_width),
                int(source_height),
            )
            self.assertEqual(width % 16, 0)
            self.assertEqual(height % 16, 0)
            self.assertLessEqual(max(width, height), 2048)
            self.assertGreaterEqual(width * height, correction.MIN_OUTPUT_PIXEL_COUNT)
            self.assertLessEqual(max(width, height) / min(width, height), 3.0)

# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
