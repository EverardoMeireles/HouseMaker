# ### Imports ###
from __future__ import annotations

import base64
import json
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import housemaker.object_reference_editing as editing
from housemaker.object_reference_editing import (
    MAX_PROMPT_CHARACTERS,
    ObjectReferenceEditingCancelled,
    ObjectReferenceEditingError,
    build_openai_object_reference_edit_multipart,
    edit_object_reference,
    openai_object_reference_edit,
)
from housemaker.qwen_plan_image_correction import (
    QwenPlanCorrectionCancelled,
    QwenPlanCorrectionError,
)

# ### Model fixtures ###
DIRECT_IMAGE_EDIT_MODELS = (
    "gpt-image-2",
    "gpt-image-2.5-sunburst",
    "gpt-image-2.5-flare",
)
EXPECTED_REFERENCE_EDIT_MODEL_OPTIONS = {
    "GPT Image 2.5 Sunburst": "gpt-image-2.5-sunburst",
    "GPT Image 2.5 Flare": "gpt-image-2.5-flare",
    "GPT Image 2": "gpt-image-2",
    "Qwen-Image-2.1": "Qwen-Image-2.1",
    "GPT-5.6 Terra": "gpt-5.6-terra",
    "GPT-5.6 Luna": "gpt-5.6-luna",
}


# ### Fixture helpers ###
def _source_bgra(width: int = 12, height: int = 4) -> np.ndarray:
    source = np.zeros((height, width, 4), dtype=np.uint8)
    source[:, :, 0] = 30
    source[:, :, 1] = 80
    source[:, :, 2] = 140
    source[:, :, 3] = 255
    source[:, :2, 3] = 0
    return source


def _png_bytes(image: np.ndarray) -> bytes:
    succeeded, encoded = cv2.imencode(".png", image)
    assert succeeded
    return encoded.tobytes()


class RecordingEditor:
    def __init__(self, color: tuple[int, int, int, int]) -> None:
        self.color = color
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        source_png: bytes,
        **kwargs: object,
    ) -> bytes:
        source = cv2.imdecode(
            np.frombuffer(source_png, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        assert source is not None
        self.calls.append(
            {
                "source": source,
                **kwargs,
            }
        )
        result = np.empty_like(source)
        result[:] = self.color
        return _png_bytes(result)


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


# ### Editing pipeline tests ###
class ObjectReferenceEditingPipelineTests(unittest.TestCase):
    def test_model_options_have_stable_labels_ids_and_default(self) -> None:
        self.assertEqual(
            dict(editing.OBJECT_REFERENCE_EDIT_MODEL_OPTIONS),
            EXPECTED_REFERENCE_EDIT_MODEL_OPTIONS,
        )
        self.assertEqual(
            len(editing.OBJECT_REFERENCE_EDIT_MODEL_OPTIONS),
            len(EXPECTED_REFERENCE_EDIT_MODEL_OPTIONS),
        )
        self.assertEqual(editing.DEFAULT_OBJECT_REFERENCE_EDIT_MODEL, "gpt-image-2")

    def test_only_local_qwen_can_edit_without_an_openai_api_key(self) -> None:
        for model in EXPECTED_REFERENCE_EDIT_MODEL_OPTIONS.values():
            with self.subTest(model=model):
                self.assertEqual(
                    editing.object_reference_edit_model_requires_api_key(model),
                    model != "Qwen-Image-2.1",
                )

    def test_letterboxes_masked_object_into_square_and_keeps_full_result(self) -> None:
        source = _source_bgra()
        editor = RecordingEditor((205, 155, 105, 77))

        result = edit_object_reference(
            source,
            "Remove the floral cover and reveal the wooden tabletop.",
            api_key="test-key",
            image_editor=editor,
        )

        self.assertEqual(len(editor.calls), 1)
        call = editor.calls[0]
        self.assertEqual(call["model"], "gpt-image-2")
        self.assertEqual(call["output_size"], (1024, 1024))
        prepared_source = call["source"]
        self.assertIsInstance(prepared_source, np.ndarray)
        assert isinstance(prepared_source, np.ndarray)
        self.assertEqual(prepared_source.shape, (1024, 1024, 4))
        self.assertTrue(np.all(prepared_source[:340, :, 3] == 0))
        self.assertTrue(np.all(prepared_source[682:, :, 3] == 0))
        self.assertGreater(np.count_nonzero(prepared_source[:, :, 3]), 0)
        transparent = prepared_source[:, :, 3] == 0
        self.assertTrue(np.all(prepared_source[transparent, :3] == 0))
        self.assertIn("Remove the floral cover", str(call["prompt"]))
        self.assertIn("complete new 1024x1024 image", str(call["prompt"]))
        self.assertIn("one isolated object", str(call["prompt"]))

        expected_replacement = np.array((205, 155, 105, 77), dtype=np.uint8)
        np.testing.assert_array_equal(
            result,
            np.broadcast_to(expected_replacement, result.shape),
        )
        self.assertEqual(result.shape, (1024, 1024, 4))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(result.flags.owndata)
        self.assertTrue(result.flags.c_contiguous)

    def test_opaque_scene_reference_is_not_described_as_an_isolated_mask(self) -> None:
        source = _source_bgra()
        source[:, :, 3] = 255
        editor = RecordingEditor((205, 155, 105, 255))

        edit_object_reference(
            source,
            "Extract and straighten the ornate cornice.",
            api_key="test-key",
            image_editor=editor,
        )

        prompt = " ".join(str(editor.calls[0]["prompt"]).lower().split())
        self.assertIn("unmasked scene photograph", prompt)
        self.assertIn("do not reproduce the room, walls, ceiling", prompt)
        self.assertIn("extract and straighten the ornate cornice", prompt)

    def test_qwen_opaque_scene_uses_scene_extraction_instructions(self) -> None:
        source = _source_bgra()
        source[:, :, 3] = 255
        editor = RecordingEditor((205, 155, 105, 255))

        edit_object_reference(
            source,
            "Extract one repeating cornice module.",
            model="Qwen-Image-2.1",
            api_key="",
            image_editor=editor,
        )

        prompt = " ".join(str(editor.calls[0]["prompt"]).lower().split())
        self.assertIn("unmasked scene photograph", prompt)
        self.assertIn("one repeating cornice module", prompt)
        self.assertNotIn("rgba image with transparency", prompt)

    def test_rejects_provider_output_that_is_not_1024_square(self) -> None:
        def wrong_size_editor(*_args: object, **_kwargs: object) -> bytes:
            return _png_bytes(np.full((512, 1024, 4), 101, dtype=np.uint8))

        with self.assertRaisesRegex(
            ObjectReferenceEditingError,
            "unexpected image size",
        ):
            edit_object_reference(
                _source_bgra(),
                "Remove the floral cover.",
                api_key="test-key",
                image_editor=wrong_size_editor,
            )

    def test_selected_model_is_forwarded_to_the_image_editor(self) -> None:
        editor = RecordingEditor((205, 155, 105, 77))

        edit_object_reference(
            _source_bgra(),
            "Remove the floral cover.",
            model="gpt-image-2.5-sunburst",
            api_key="test-key",
            image_editor=editor,
        )

        self.assertEqual(len(editor.calls), 1)
        self.assertEqual(editor.calls[0]["model"], "gpt-image-2.5-sunburst")

    def test_qwen_reference_has_transparent_outpainting_room(self) -> None:
        editor = RecordingEditor((205, 155, 105, 77))

        edit_object_reference(
            _source_bgra(),
            "Remove the floral cover.",
            model="Qwen-Image-2.1",
            api_key="",
            image_editor=editor,
        )

        prepared_source = editor.calls[0]["source"]
        assert isinstance(prepared_source, np.ndarray)
        _visible_rows, visible_columns = np.nonzero(
            prepared_source[:, :, 3] > 0
        )
        self.assertLessEqual(
            int(visible_columns.max() - visible_columns.min() + 1),
            666,
        )
        self.assertGreater(int(visible_columns.min()), 0)
        self.assertLess(int(visible_columns.max()), 1023)

    def test_responses_and_qwen_models_dispatch_to_their_adapters(self) -> None:
        providers = (
            (
                "gpt-5.6-terra",
                "openai_responses_object_reference_edit",
                "test-key",
            ),
            (
                "gpt-5.6-luna",
                "openai_responses_object_reference_edit",
                "test-key",
            ),
            (
                "Qwen-Image-2.1",
                "qwen_object_reference_edit",
                "",
            ),
        )
        for model, adapter_name, api_key in providers:
            with self.subTest(model=model):
                editor = RecordingEditor((205, 155, 105, 77))
                with patch.object(editing, adapter_name, editor):
                    result = edit_object_reference(
                        _source_bgra(),
                        "Remove the floral cover.",
                        model=model,
                        api_key=api_key,
                    )

                self.assertEqual(len(editor.calls), 1)
                self.assertEqual(editor.calls[0]["model"], model)
                self.assertEqual(editor.calls[0]["api_key"], api_key)
                self.assertEqual(result.shape, (1024, 1024, 4))
                self.assertTrue(np.all(result == (205, 155, 105, 77)))

    def test_responses_models_request_a_plain_neutral_opaque_background(
        self,
    ) -> None:
        for model in ("gpt-5.6-terra", "gpt-5.6-luna"):
            with self.subTest(model=model):
                editor = RecordingEditor((205, 155, 105, 255))

                edit_object_reference(
                    _source_bgra(),
                    "Remove the floral cover.",
                    model=model,
                    api_key="test-key",
                    image_editor=editor,
                )

                prompt = " ".join(
                    str(editor.calls[0]["prompt"]).lower().split()
                )
                self.assertIn("plain neutral opaque background", prompt)
                self.assertNotIn("transparent background", prompt)

    def test_rejects_an_unsupported_model_before_calling_the_editor(self) -> None:
        editor = RecordingEditor((205, 155, 105, 77))

        with self.assertRaisesRegex(
            ObjectReferenceEditingError,
            "supported object reference editing model",
        ):
            edit_object_reference(
                _source_bgra(),
                "Remove the floral cover.",
                model="not-an-image-model",
                api_key="test-key",
                image_editor=editor,
            )

        self.assertEqual(editor.calls, [])

    def test_rejects_empty_alpha(self) -> None:
        transparent = _source_bgra()
        transparent[:, :, 3] = 0
        with self.assertRaisesRegex(
            ObjectReferenceEditingError,
            "no visible subject",
        ):
            edit_object_reference(
                transparent,
                "Remove cover",
                api_key="test-key",
                image_editor=RecordingEditor((1, 2, 3, 4)),
            )

    def test_rejects_non_bgra_source(self) -> None:
        with self.assertRaisesRegex(ObjectReferenceEditingError, "uint8 BGRA"):
            edit_object_reference(
                np.ones((2, 2, 3), dtype=np.uint8),
                "Remove cover",
                api_key="test-key",
                image_editor=RecordingEditor((1, 2, 3, 4)),
            )

    def test_prompt_must_be_nonblank_and_bounded(self) -> None:
        source = _source_bgra()
        for prompt in ("   ", "x" * (MAX_PROMPT_CHARACTERS + 1)):
            with (
                self.subTest(length=len(prompt)),
                self.assertRaises(ObjectReferenceEditingError),
            ):
                edit_object_reference(
                    source,
                    prompt,
                    api_key="test-key",
                    image_editor=RecordingEditor((1, 2, 3, 4)),
                )

    def test_provider_error_does_not_expose_key_or_provider_message(self) -> None:
        api_key = "sk-secret-reference-key"

        def failing_editor(*_args: object, **_kwargs: object) -> bytes:
            raise RuntimeError(f"provider refused {api_key}: private response")

        with self.assertRaises(ObjectReferenceEditingError) as raised:
            edit_object_reference(
                _source_bgra(),
                "Remove cover",
                api_key=api_key,
                image_editor=failing_editor,
            )
        message = str(raised.exception)
        self.assertNotIn(api_key, message)
        self.assertNotIn("private response", message)
        self.assertEqual(message, "The AI object reference edit failed.")

    def test_cancellation_returns_without_waiting_for_stalled_editor(self) -> None:
        started = threading.Event()
        released = threading.Event()

        def blocking_editor(
            *_args: object,
            cancellation_check=None,
            **_kwargs: object,
        ) -> bytes:
            started.set()
            while cancellation_check is None or not cancellation_check():
                time.sleep(0.005)
            released.set()
            return b"unused"

        checks = 0

        def cancel_after_start() -> bool:
            nonlocal checks
            checks += 1
            return started.is_set() and checks > 1

        with self.assertRaises(ObjectReferenceEditingCancelled):
            edit_object_reference(
                _source_bgra(),
                "Remove cover",
                api_key="test-key",
                cancellation_check=cancel_after_start,
                image_editor=blocking_editor,
            )
        self.assertTrue(released.wait(timeout=1.0))


# ### Multipart and transport tests ###
class OpenAIObjectReferenceEditingTests(unittest.TestCase):
    def test_multipart_contains_square_generation_fields_and_only_source_png(
        self,
    ) -> None:
        source_png = _png_bytes(np.full((2, 3, 4), 91, dtype=np.uint8))
        body, content_type = build_openai_object_reference_edit_multipart(
            source_png,
            "Remove the cover",
            output_size=(1024, 1024),
        )

        self.assertTrue(content_type.startswith("multipart/form-data; boundary="))
        self.assertIn(b'name="model"', body)
        self.assertIn(b"gpt-image-2", body)
        self.assertIn(b'name="image"; filename="object_reference.png"', body)
        self.assertNotIn(b'name="mask"', body)
        self.assertIn(b'name="quality"\r\n\r\nhigh', body)
        self.assertIn(b'name="background"\r\n\r\ntransparent', body)
        self.assertIn(b'name="output_format"\r\n\r\npng', body)
        self.assertIn(b'name="size"\r\n\r\n1024x1024', body)
        self.assertIn(source_png, body)

    def test_multipart_uses_each_selected_supported_model(self) -> None:
        source_png = _png_bytes(np.full((2, 3, 4), 91, dtype=np.uint8))

        for model in DIRECT_IMAGE_EDIT_MODELS:
            with self.subTest(model=model):
                body, _content_type = (
                    build_openai_object_reference_edit_multipart(
                        source_png,
                        "Remove the cover",
                        model=model,
                        output_size=(1024, 1024),
                    )
                )

                self.assertIn(
                    b'name="model"\r\n\r\n' + model.encode("ascii"),
                    body,
                )

    def test_multipart_rejects_models_that_do_not_use_images_edits(self) -> None:
        source_png = _png_bytes(np.full((2, 3, 4), 91, dtype=np.uint8))

        for model in ("gpt-5.6-terra", "gpt-5.6-luna", "Qwen-Image-2.1"):
            with (
                self.subTest(model=model),
                self.assertRaises(ObjectReferenceEditingError),
            ):
                build_openai_object_reference_edit_multipart(
                    source_png,
                    "Remove the cover",
                    model=model,
                    output_size=(1024, 1024),
                )

    def test_transport_posts_to_images_edits_and_decodes_base64_png(self) -> None:
        source_png = _png_bytes(np.full((2, 3, 4), 20, dtype=np.uint8))
        output_png = _png_bytes(np.full((4, 4, 4), 120, dtype=np.uint8))
        response_payload = json.dumps(
            {
                "data": [
                    {"b64_json": base64.b64encode(output_png).decode("ascii")}
                ]
            }
        ).encode("utf-8")
        captured = {}

        def opener(request, **kwargs):
            captured["request"] = request
            captured["kwargs"] = kwargs
            return FakeResponse(response_payload)

        result = openai_object_reference_edit(
            source_png,
            model="gpt-image-2.5-flare",
            api_key="test-key",
            prompt="Remove the cover",
            output_size=(1024, 1024),
            cancellation_check=None,
            opener=opener,
        )

        self.assertEqual(result, output_png)
        request = captured["request"]
        self.assertEqual(request.full_url, editing.OPENAI_IMAGE_EDIT_URL)
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertIn("multipart/form-data", request.get_header("Content-type"))
        self.assertIn(
            b'name="model"\r\n\r\ngpt-image-2.5-flare',
            request.data,
        )
        self.assertEqual(
            captured["kwargs"]["timeout"],
            editing.OPENAI_NETWORK_TIMEOUT_SECONDS,
        )

    def test_transport_redacts_network_and_provider_failures(self) -> None:
        api_key = "sk-never-show-this"

        def opener(_request, **_kwargs):
            raise OSError(f"network response leaked {api_key}")

        png = _png_bytes(np.full((1, 1, 4), 255, dtype=np.uint8))
        with self.assertRaises(ObjectReferenceEditingError) as raised:
            openai_object_reference_edit(
                png,
                api_key=api_key,
                prompt="Remove cover",
                output_size=(1024, 1024),
                cancellation_check=None,
                opener=opener,
            )
        self.assertNotIn(api_key, str(raised.exception))
        self.assertNotIn("network response", str(raised.exception))


# ### Responses and local-Qwen adapter tests ###
class AlternateObjectReferenceEditingAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_png = _png_bytes(
            np.full((1024, 1024, 4), 91, dtype=np.uint8)
        )
        self.output_png = _png_bytes(
            np.full((1024, 1024, 4), 211, dtype=np.uint8)
        )

    def test_responses_adapter_forwards_each_gpt_5_6_model(self) -> None:
        calls: list[tuple[bytes, dict[str, object]]] = []

        def responses_editor(
            image_bytes: bytes,
            **kwargs: object,
        ) -> bytes:
            calls.append((image_bytes, kwargs))
            return self.output_png

        for model in ("gpt-5.6-terra", "gpt-5.6-luna"):
            with self.subTest(model=model):
                calls.clear()
                with patch.object(
                    editing,
                    "openai_responses_image_edit",
                    responses_editor,
                ):
                    result = editing.openai_responses_object_reference_edit(
                        self.source_png,
                        api_key="test-key",
                        model=model,
                        prompt="Remove the cover",
                        output_size=(1024, 1024),
                        cancellation_check=None,
                    )

                self.assertEqual(result, self.output_png)
                self.assertEqual(len(calls), 1)
                image_bytes, arguments = calls[0]
                self.assertEqual(image_bytes, self.source_png)
                self.assertEqual(arguments["model"], model)
                self.assertEqual(arguments["api_key"], "test-key")
                self.assertEqual(arguments["prompt"], "Remove the cover")
                self.assertEqual(arguments["output_size"], (1024, 1024))
                self.assertIsNone(arguments["cancellation_check"])
                self.assertEqual(arguments["background"], "opaque")
                self.assertEqual(arguments["action"], "generate")

    def test_qwen_adapter_uses_local_factory_without_api_key(self) -> None:
        calls: list[tuple[bytes, dict[str, object]]] = []

        def local_editor(image_bytes: bytes, **kwargs: object) -> bytes:
            calls.append((image_bytes, kwargs))
            return self.output_png

        with patch.object(
            editing,
            "create_default_qwen_object_reference_editor",
            return_value=local_editor,
        ) as factory:
            result = editing.qwen_object_reference_edit(
                self.source_png,
                api_key="",
                model="Qwen-Image-2.1",
                prompt="Remove the cover",
                output_size=(1024, 1024),
                cancellation_check=None,
            )

        self.assertEqual(result, self.output_png)
        factory.assert_called_once_with()
        self.assertEqual(len(calls), 1)
        image_bytes, arguments = calls[0]
        self.assertEqual(image_bytes, self.source_png)
        self.assertEqual(arguments["api_key"], "")
        self.assertIn("Remove the cover", str(arguments["prompt"]))
        self.assertIn("not as the\nrequested output boundary", str(arguments["prompt"]))
        self.assertIn("genuinely new 1024x1024 composition", str(arguments["prompt"]))
        self.assertIn("RGBA image with transparency", str(arguments["prompt"]))
        self.assertEqual(arguments["output_size"], (1024, 1024))
        self.assertIsNone(arguments["cancellation_check"])

    def test_qwen_adapter_preserves_safe_local_errors(self) -> None:
        def failing_editor(*_args: object, **_kwargs: object) -> bytes:
            raise QwenPlanCorrectionError("The local Qwen weights are unavailable.")

        with (
            patch.object(
                editing,
                "create_default_qwen_object_reference_editor",
                return_value=failing_editor,
            ),
            self.assertRaisesRegex(
                ObjectReferenceEditingError,
                "local Qwen weights are unavailable",
            ),
        ):
            editing.qwen_object_reference_edit(
                self.source_png,
                api_key="",
                model="Qwen-Image-2.1",
                prompt="Remove the cover",
                output_size=(1024, 1024),
                cancellation_check=None,
            )

    def test_qwen_adapter_translates_local_cancellation(self) -> None:
        def cancelled_editor(*_args: object, **_kwargs: object) -> bytes:
            raise QwenPlanCorrectionCancelled("cancelled")

        with (
            patch.object(
                editing,
                "create_default_qwen_object_reference_editor",
                return_value=cancelled_editor,
            ),
            self.assertRaises(ObjectReferenceEditingCancelled),
        ):
            editing.qwen_object_reference_edit(
                self.source_png,
                api_key="",
                model="Qwen-Image-2.1",
                prompt="Remove the cover",
                output_size=(1024, 1024),
                cancellation_check=None,
            )


if __name__ == "__main__":
    unittest.main()
