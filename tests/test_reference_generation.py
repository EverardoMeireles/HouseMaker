"""Reference regeneration tests with no network requests or model downloads."""

# ### Imports ###
from __future__ import annotations

import threading
import time
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from housemaker import reference_generation as reference
from housemaker.plan_correction_models import (
    OPENAI_PLAN_CORRECTION_MODELS,
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
)
from housemaker.plan_image_correction import (
    PlanImageCorrectionCancelled,
    PlanImageCorrectionError,
)
from housemaker.qwen_plan_image_correction import (
    QwenPlanCorrectionCancelled,
    QwenPlanCorrectionError,
)
from housemaker.reference_generation_models import (
    DEFAULT_REFERENCE_NEGATIVE_PROMPT,
    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
)
from housemaker.stable_diffusion_reference import (
    StableDiffusionReferenceCancelled,
    StableDiffusionReferenceError,
)


# ### Image fixtures ###
def _masked_source() -> np.ndarray:
    image = np.zeros((30, 50, 4), dtype=np.uint8)
    image[:, :, :3] = (200, 40, 220)
    image[10:20, 15:35] = (30, 80, 190, 255)
    return image


def _png(image: Image.Image | None = None) -> bytes:
    image = image or Image.new("RGBA", (1024, 1024), (20, 80, 190, 255))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _decoded_rgb(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB")).copy()


# ### Input isolation and fitting ###
class ReferenceInputPreparationTests(unittest.TestCase):
    def test_isolates_only_masked_pixels_on_a_centered_white_square(self) -> None:
        source = _masked_source()
        original = source.copy()

        payload = reference.prepare_reference_generation_input(source)

        with Image.open(BytesIO(payload)) as image:
            self.assertEqual(image.size, (1024, 1024))
            self.assertEqual(image.mode, "RGB")
        prepared = _decoded_rgb(payload)
        non_white = np.any(prepared != 255, axis=2)
        rows, columns = np.nonzero(non_white)
        self.assertEqual((columns.min(), columns.max()), (82, 941))
        self.assertEqual((rows.min(), rows.max()), (297, 726))
        np.testing.assert_array_equal(prepared[512, 512], (190, 80, 30))
        np.testing.assert_array_equal(prepared[0, 0], (255, 255, 255))
        np.testing.assert_array_equal(source, original)

    def test_invisible_pixels_outside_the_mask_never_affect_the_reference(self) -> None:
        first = _masked_source()
        second = first.copy()
        second[first[:, :, 3] == 0, :3] = (0, 255, 0)

        self.assertEqual(
            reference.prepare_reference_generation_input(first),
            reference.prepare_reference_generation_input(second),
        )

    def test_translucent_mask_is_composited_on_white(self) -> None:
        source = np.full((4, 4, 4), (0, 0, 0, 128), dtype=np.uint8)

        prepared = _decoded_rgb(reference.prepare_reference_generation_input(source))

        np.testing.assert_array_equal(prepared[512, 512], (127, 127, 127))

    def test_numeric_pixels_are_clamped_without_mutating_the_input(self) -> None:
        source = np.full((3, 4, 4), (500.0, -20.0, 100.0, 300.0))
        original = source.copy()

        prepared = _decoded_rgb(reference.prepare_reference_generation_input(source))

        np.testing.assert_array_equal(prepared[512, 512], (100, 0, 255))
        np.testing.assert_array_equal(source, original)

    def test_rejects_empty_invalid_and_fully_transparent_masks(self) -> None:
        invalid_images = (
            None,
            np.empty((0, 0, 4), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 4), dtype=np.uint8),
            np.full((4, 4, 4), np.nan),
            np.full((4, 4, 4), np.inf),
            np.full((4, 4, 4), "not a pixel"),
        )
        for source in invalid_images:
            with (
                self.subTest(shape=getattr(source, "shape", None)),
                self.assertRaises(reference.ReferenceGenerationError),
            ):
                reference.prepare_reference_generation_input(source)

    def test_rejects_oversized_input_before_processing_it(self) -> None:
        with (
            patch.object(reference, "MAX_REFERENCE_PIXELS", 4),
            self.assertRaisesRegex(reference.ReferenceGenerationError, "too large"),
        ):
            reference.prepare_reference_generation_input(_masked_source())


# ### Generated image decoding ###
class GeneratedReferenceDecodingTests(unittest.TestCase):
    def test_preserves_generated_alpha_instead_of_reusing_the_original_mask(
        self,
    ) -> None:
        generated = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        generated.paste((30, 100, 200, 190), (100, 400, 900, 600))

        result = reference._decode_generated_reference(_png(generated))

        self.assertEqual(result.shape, (1024, 1024, 4))
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result[450, 150], (200, 100, 30, 190))
        self.assertEqual(int(result[300, 512, 3]), 0)
        rows, columns = np.nonzero(result[:, :, 3])
        self.assertEqual((columns.min(), columns.max()), (100, 899))
        self.assertEqual((rows.min(), rows.max()), (400, 599))

    def test_rectangular_output_is_fitted_without_stretching(self) -> None:
        payload = _png(Image.new("RGB", (100, 50), (10, 40, 90)))

        result = reference._decode_generated_reference(payload)

        self.assertEqual(result.shape, (1024, 1024, 4))
        self.assertTrue(np.all(result[256:768, :, 3] == 255))
        self.assertTrue(np.all(result[:256, :, 3] == 0))
        self.assertTrue(np.all(result[768:, :, 3] == 0))
        np.testing.assert_array_equal(result[512, 512], (90, 40, 10, 255))

    def test_rejects_missing_corrupt_and_fully_transparent_outputs(self) -> None:
        invalid_payloads = (
            None,
            b"",
            b"not an image",
            _png(Image.new("RGBA", (20, 20), (0, 0, 0, 0))),
        )
        for payload in invalid_payloads:
            with (
                self.subTest(payload_type=type(payload)),
                self.assertRaises(reference.ReferenceGenerationError),
            ):
                reference._decode_generated_reference(payload)

    def test_rejects_oversized_generated_payload_and_image(self) -> None:
        payload = _png(Image.new("RGB", (4, 4), "white"))
        with (
            patch.object(reference, "MAX_REFERENCE_BYTES", 1),
            self.assertRaises(reference.ReferenceGenerationError),
        ):
            reference._decode_generated_reference(payload)
        with (
            patch.object(reference, "MAX_REFERENCE_PIXELS", 4),
            self.assertRaisesRegex(reference.ReferenceGenerationError, "too large"),
        ):
            reference._decode_generated_reference(payload)


# ### Prompt and selected-provider routing ###
class ReferenceGenerationProviderTests(unittest.TestCase):
    def test_instruction_models_request_tight_framing_without_changing_proportions(
        self,
    ) -> None:
        instruction = "Show the same object directly from the front."
        for model in (
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
        ):
            with self.subTest(model=model):
                prompt = reference.build_reference_generation_prompt(
                    instruction, model=model
                )

                self.assertIn("longest dimension", prompt)
                self.assertIn("approximately 90-95% of the canvas", prompt)
                self.assertIn("thin safety margin", prompt)
                self.assertIn("entire object centered", prompt)
                self.assertIn("Do not crop any part", prompt)
                self.assertIn("including all components and protrusions", prompt)
                self.assertIn("natural aspect ratio", prompt)
                self.assertIn("do not stretch it to fill both image dimensions", prompt)
                self.assertIn("Do not copy excess white padding", prompt)
                self.assertIn(
                    "viewpoint unless the user explicitly requests otherwise", prompt
                )
                self.assertTrue(prompt.endswith(instruction))

    def test_instruction_models_use_the_same_object_agnostic_guidance(self) -> None:
        instruction = "Improve clarity without redesigning the object."
        model_prompts = {
            reference.build_reference_generation_prompt(instruction, model=model)
            for model in (
                PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
                PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
                PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
                PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
            )
        }

        self.assertEqual(len(model_prompts), 1)
        prompt = model_prompts.pop()
        guidance, requested_changes = prompt.split("User's requested object changes:\n")
        self.assertEqual(requested_changes, instruction)
        self.assertIn(
            "shape, proportions, materials, colors, surface details and component arrangement",
            guidance,
        )
        self.assertIn("painted-mask outline", guidance)
        self.assertIn("use the object's actual outline, not the selection shape", guidance)
        self.assertIn(
            "Improve visual clarity without redesigning the object or inventing extra components",
            guidance,
        )
        self.assertIn("except where the user explicitly requests a change", guidance)
        self.assertIn("unless the user explicitly requests otherwise", guidance)
        for object_specific_term in ("cupboard", "molding", "feet", "handles"):
            with self.subTest(term=object_specific_term):
                self.assertNotIn(object_specific_term, guidance.lower())

    def test_custom_object_changes_are_kept_separate_and_unchanged(self) -> None:
        instruction = (
            "Remove the floral cover from the table.\n"
            "Show its uncovered wooden surface from directly above."
        )

        prompt = reference.build_reference_generation_prompt(
            instruction, model=PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1
        )

        guidance, requested_changes = prompt.split("User's requested object changes:\n")
        self.assertEqual(requested_changes, instruction)
        self.assertNotIn("floral cover", guidance)
        self.assertNotIn("wooden surface", guidance)

    def test_sdxl_keeps_its_descriptive_prompt_without_chat_framing_instructions(
        self,
    ) -> None:
        instruction = "A dark wooden cupboard, straight-on front view"

        prompt = reference.build_reference_generation_prompt(
            instruction, model=REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION
        )

        self.assertEqual(
            prompt,
            instruction + ", single isolated object, clean white background, "
            "complete object in frame, coherent geometry, realistic materials, "
            "sharp details, even studio lighting",
        )
        self.assertNotIn("90-95%", prompt)
        self.assertNotIn("User's requested", prompt)

    def test_api_request_uses_selected_model_prepared_crop_and_object_prompt(
        self,
    ) -> None:
        editor = Mock(return_value=_png())
        progress = Mock()
        with patch.object(
            reference, "_create_reference_image_editor", return_value=editor
        ) as create:
            result = reference.generate_object_reference(
                _masked_source(),
                "remove the floral tablecloth",
                PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
                api_key=" test-key ",
                negative_prompt="side view, irregular cutout border",
                progress_callback=progress,
            )

        create.assert_called_once_with(PLAN_CORRECTION_MODEL_GPT_IMAGE_2)
        editor.assert_called_once()
        sent = editor.call_args
        self.assertEqual(sent.kwargs["api_key"], "test-key")
        self.assertEqual(sent.kwargs["output_size"], (1024, 1024))
        self.assertNotIn("negative_prompt", sent.kwargs)
        self.assertNotIn("irregular cutout border", sent.kwargs["prompt"])
        self.assertIn("NEW standalone 1024x1024", sent.kwargs["prompt"])
        self.assertIn("remove the floral tablecloth", sent.kwargs["prompt"])
        self.assertNotIn("floor plan", sent.kwargs["prompt"].lower())
        self.assertEqual(
            sent.args[0],
            reference.prepare_reference_generation_input(_masked_source()),
        )
        self.assertEqual(result.shape, (1024, 1024, 4))
        self.assertEqual(progress.call_args.args, ("Reference ready", 100))

    def test_luna_and_terra_request_new_images_with_opaque_backgrounds(self) -> None:
        for model in (
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        ):
            with self.subTest(model=model):
                with patch(
                    "housemaker.plan_image_correction.openai_responses_image_edit",
                    return_value=_png(),
                ) as edit:
                    reference.generate_object_reference(
                        _masked_source(),
                        "a plain wooden table",
                        model,
                        api_key="test-key",
                    )

                edit.assert_called_once()
                self.assertEqual(edit.call_args.kwargs["model"], model)
                self.assertEqual(edit.call_args.kwargs["action"], "generate")
                self.assertEqual(edit.call_args.kwargs["background"], "opaque")

    def test_local_qwen_needs_no_api_key_and_skips_plan_correction(self) -> None:
        editor = Mock(return_value=_png())
        with (
            patch(
                "housemaker.qwen_plan_image_correction.create_default_qwen_plan_image_editor",
                return_value=editor,
            ) as create,
            patch(
                "housemaker.plan_image_correction.correct_plan_image"
            ) as correct_plan,
        ):
            reference.generate_object_reference(
                _masked_source(), "a wooden chair", PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1
            )

        create.assert_called_once_with()
        editor.assert_called_once()
        self.assertEqual(editor.call_args.kwargs["api_key"], "")
        self.assertEqual(editor.call_args.kwargs["output_size"], (1024, 1024))
        self.assertIn(
            "approximately 90-95% of the canvas", editor.call_args.kwargs["prompt"]
        )
        self.assertIn("a wooden chair", editor.call_args.kwargs["prompt"])
        correct_plan.assert_not_called()

    def test_stable_diffusion_dispatch_forwards_strength_progress_and_cancel(
        self,
    ) -> None:
        cancelled = Mock(return_value=False)
        progress = Mock()
        with (
            patch(
                "housemaker.stable_diffusion_reference.generate_stable_diffusion_reference",
                return_value=_png(),
            ) as generate_sd,
            patch.object(reference, "_create_reference_image_editor") as create,
            patch(
                "housemaker.plan_image_correction.correct_plan_image"
            ) as correct_plan,
        ):
            reference.generate_object_reference(
                _masked_source(),
                "a carved wooden chair",
                REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
                strength=0.65,
                negative_prompt="side view, mask outline",
                progress_callback=progress,
                cancellation_check=cancelled,
            )

        generate_sd.assert_called_once()
        sent = generate_sd.call_args
        self.assertEqual(sent.kwargs["strength"], 0.65)
        self.assertEqual(sent.kwargs["negative_prompt"], "side view, mask outline")
        self.assertEqual(sent.kwargs["output_size"], (1024, 1024))
        self.assertIs(sent.kwargs["cancellation_check"], cancelled)
        self.assertIs(sent.kwargs["progress_callback"], progress)
        self.assertIn("a carved wooden chair", sent.kwargs["prompt"])
        self.assertIn("single isolated object", sent.kwargs["prompt"])
        self.assertNotIn("User's requested", sent.kwargs["prompt"])
        create.assert_not_called()
        correct_plan.assert_not_called()

    def test_sdxl_negative_prompt_defaults_and_can_be_cleared(self) -> None:
        for options, expected in (
            ({}, DEFAULT_REFERENCE_NEGATIVE_PROMPT),
            ({"negative_prompt": ""}, ""),
        ):
            with (
                self.subTest(options=options),
                patch(
                    "housemaker.stable_diffusion_reference.generate_stable_diffusion_reference",
                    return_value=_png(),
                ) as generate_sd,
            ):
                reference.generate_object_reference(
                    _masked_source(),
                    "a wooden cupboard, front view",
                    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
                    **options,
                )
            self.assertEqual(
                generate_sd.call_args.kwargs["negative_prompt"], expected
            )

    def test_missing_api_keys_are_rejected_only_for_paid_models(self) -> None:
        for model in OPENAI_PLAN_CORRECTION_MODELS:
            with self.subTest(model=model):
                with (
                    patch.object(reference, "_create_reference_image_editor") as create,
                    self.assertRaisesRegex(
                        reference.ReferenceGenerationError, "API key"
                    ),
                ):
                    reference.generate_object_reference(
                        _masked_source(), "a chair", model, api_key=" "
                    )
                create.assert_not_called()

    def test_invalid_models_and_prompts_do_not_start_a_provider(self) -> None:
        for model, prompt in (
            ("unsupported-model", "a chair"),
            (PLAN_CORRECTION_MODEL_GPT_IMAGE_2, " "),
            (PLAN_CORRECTION_MODEL_GPT_IMAGE_2, "x" * 4001),
        ):
            with self.subTest(model=model, prompt_length=len(prompt)):
                with (
                    patch.object(reference, "_create_reference_image_editor") as create,
                    self.assertRaises(reference.ReferenceGenerationError),
                ):
                    reference.generate_object_reference(
                        _masked_source(), prompt, model, api_key="test-key"
                    )
                create.assert_not_called()

    def test_provider_failures_do_not_trigger_fallbacks(self) -> None:
        for failure in (
            PlanImageCorrectionError("provider unavailable"),
            QwenPlanCorrectionError("local runtime failed"),
        ):
            with self.subTest(failure=type(failure)):
                with (
                    patch.object(
                        reference,
                        "_create_reference_image_editor",
                        return_value=Mock(side_effect=failure),
                    ) as create,
                    patch(
                        "housemaker.stable_diffusion_reference.generate_stable_diffusion_reference"
                    ) as generate_sd,
                    self.assertRaisesRegex(
                        reference.ReferenceGenerationError, str(failure)
                    ),
                ):
                    reference.generate_object_reference(
                        _masked_source(),
                        "a chair",
                        PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
                        api_key="test-key",
                    )
                create.assert_called_once()
                generate_sd.assert_not_called()

    def test_local_stable_diffusion_failures_and_cancellation_are_translated(
        self,
    ) -> None:
        cases = (
            (
                StableDiffusionReferenceError("model missing"),
                reference.ReferenceGenerationError,
            ),
            (
                StableDiffusionReferenceCancelled("cancelled"),
                reference.ReferenceGenerationCancelled,
            ),
        )
        for failure, expected in cases:
            with (
                self.subTest(failure=type(failure)),
                patch(
                    "housemaker.stable_diffusion_reference.generate_stable_diffusion_reference",
                    side_effect=failure,
                ),
                self.assertRaises(expected),
            ):
                reference.generate_object_reference(
                    _masked_source(),
                    "a chair",
                    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
                )


# ### Cooperative cancellation ###
class ReferenceGenerationCancellationTests(unittest.TestCase):
    def test_pre_cancelled_request_starts_nothing(self) -> None:
        with (
            patch.object(reference, "_create_reference_image_editor") as create,
            self.assertRaises(reference.ReferenceGenerationCancelled),
        ):
            reference.generate_object_reference(
                _masked_source(),
                "a chair",
                PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
                api_key="test-key",
                cancellation_check=lambda: True,
            )
        create.assert_not_called()

    def test_provider_specific_cancellation_preserves_public_cancelled_error(
        self,
    ) -> None:
        for failure in (
            PlanImageCorrectionCancelled("cancelled"),
            QwenPlanCorrectionCancelled("cancelled"),
        ):
            with (
                self.subTest(failure=type(failure)),
                patch.object(
                    reference,
                    "_create_reference_image_editor",
                    return_value=Mock(side_effect=failure),
                ),
                self.assertRaises(reference.ReferenceGenerationCancelled),
            ):
                reference.generate_object_reference(
                    _masked_source(),
                    "a chair",
                    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
                    api_key="test-key",
                )

    def test_cancelling_a_stalled_api_editor_does_not_wait_for_it(self) -> None:
        cancelled = threading.Event()
        release_editor = threading.Event()
        transport_cancelled = []

        def stalled_editor(*_args: object, **kwargs: object) -> bytes:
            transport_cancelled.append(kwargs["cancellation_check"])
            cancelled.set()
            release_editor.wait(2.0)
            return _png()

        started = time.monotonic()
        try:
            with (
                patch.object(
                    reference,
                    "_create_reference_image_editor",
                    return_value=stalled_editor,
                ),
                self.assertRaises(reference.ReferenceGenerationCancelled),
            ):
                reference.generate_object_reference(
                    _masked_source(),
                    "a chair",
                    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
                    api_key="test-key",
                    cancellation_check=cancelled.is_set,
                )
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertTrue(transport_cancelled[0]())
        finally:
            release_editor.set()

    def test_cancelled_completed_output_is_never_returned_as_success(self) -> None:
        cancelled = threading.Event()

        def editor(*_args: object, **_kwargs: object) -> bytes:
            cancelled.set()
            return _png()

        with (
            patch.object(
                reference, "_create_reference_image_editor", return_value=editor
            ),
            self.assertRaises(reference.ReferenceGenerationCancelled),
        ):
            reference.generate_object_reference(
                _masked_source(),
                "a chair",
                PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
                api_key="test-key",
                cancellation_check=cancelled.is_set,
            )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
