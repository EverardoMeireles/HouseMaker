# ### Imports ###
from __future__ import annotations

import base64
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from housemaker import plan_image_correction as correction
from housemaker import reference_generation as reference
from housemaker import surface_texture_providers as surfaces
from housemaker.openai_batch import (
    LUNA_BATCH_MODEL,
    LUNA_MODEL,
    OpenAIBatchCancelled,
    OpenAIBatchError,
)
from housemaker.plan_correction_models import (
    OPENAI_PLAN_CORRECTION_MODELS,
    PLAN_CORRECTION_MODEL_OPTIONS,
)
from housemaker.reference_generation_models import REFERENCE_GENERATION_MODEL_OPTIONS


# ### Fixtures ###
def _png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (1024, 1024), (30, 70, 120)).save(stream, format="PNG")
    return stream.getvalue()


def _result(payload: bytes) -> dict:
    return {
        "output": [
            {
                "type": "image_generation_call",
                "result": base64.b64encode(payload).decode("ascii"),
            }
        ]
    }


# ### Image and option routing ###
class LunaBatchImageWorkflowTests(unittest.TestCase):
    def test_batch_is_an_additional_paid_plan_and_reference_option(self) -> None:
        for options in (
            PLAN_CORRECTION_MODEL_OPTIONS,
            REFERENCE_GENERATION_MODEL_OPTIONS,
        ):
            ids = [model for _label, model in options]
            self.assertEqual(ids[ids.index(LUNA_MODEL) + 1], LUNA_BATCH_MODEL)
        self.assertIn(LUNA_BATCH_MODEL, OPENAI_PLAN_CORRECTION_MODELS)

    def test_image_body_maps_only_internal_batch_alias(self) -> None:
        body = correction.build_openai_responses_image_edit_body(
            b"input",
            model=LUNA_BATCH_MODEL,
            prompt="empty shelves",
            action="generate",
            background="opaque",
        )
        self.assertEqual(body["model"], LUNA_MODEL)
        self.assertEqual(body["input"][0]["content"][0]["text"], "empty shelves")
        self.assertEqual(body["tools"][0]["background"], "opaque")
        self.assertEqual(body["tools"][0]["action"], "generate")
        self.assertEqual(body["tools"][0]["size"], "1024x1024")
        self.assertNotIn("service_tier", body)

    def test_image_batch_transport_decodes_result_and_propagates_progress(self) -> None:
        png = _png()
        updates = []
        cancelled = lambda: False

        def batch(body, **kwargs):
            self.assertEqual(body["model"], LUNA_MODEL)
            self.assertIs(kwargs["cancellation_check"], cancelled)
            kwargs["progress_callback"]("Queued (24-hour completion window)")
            return _result(png)

        with (
            patch.object(correction, "request_luna_batch_response", side_effect=batch),
            patch.object(
                correction,
                "urlopen",
                side_effect=AssertionError("No realtime fallback"),
            ),
        ):
            result = correction.openai_responses_image_edit(
                b"input",
                model=LUNA_BATCH_MODEL,
                api_key="key",
                prompt="prompt",
                output_size=(1024, 1024),
                cancellation_check=cancelled,
                progress_callback=lambda message, percent: updates.append(
                    (message, percent)
                ),
            )
        self.assertEqual(result, png)
        self.assertIn("24-hour", updates[0][0])

    def test_image_batch_failures_never_fall_back_to_standard(self) -> None:
        for error, expected in (
            (
                OpenAIBatchCancelled("Batch cancelled"),
                correction.PlanImageCorrectionCancelled,
            ),
            (
                OpenAIBatchError("Batch expired"),
                correction.PlanImageCorrectionInferenceError,
            ),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(
                    correction, "request_luna_batch_response", side_effect=error
                ),
                patch.object(
                    correction,
                    "urlopen",
                    side_effect=AssertionError("No realtime fallback"),
                ),
                self.assertRaisesRegex(expected, "Batch"),
            ):
                correction.openai_responses_image_edit(
                    b"input",
                    model=LUNA_BATCH_MODEL,
                    api_key="key",
                    prompt="prompt",
                    output_size=(1024, 1024),
                    cancellation_check=None,
                )

    def test_reference_uses_batch_generation_with_opaque_background(self) -> None:
        updates = []

        def batch(body, **kwargs):
            self.assertEqual(body["model"], LUNA_MODEL)
            self.assertEqual(body["tools"][0]["action"], "generate")
            self.assertEqual(body["tools"][0]["background"], "opaque")
            kwargs["progress_callback"]("Queued for up to 24 hours")
            return _result(_png())

        with patch.object(correction, "request_luna_batch_response", side_effect=batch):
            output = reference.generate_object_reference(
                np.full((24, 32, 4), 255, dtype=np.uint8),
                "front view",
                LUNA_BATCH_MODEL,
                api_key="key",
                progress_callback=lambda message, percent: updates.append(
                    (message, percent)
                ),
            )
        self.assertEqual(output.shape, (1024, 1024, 4))
        self.assertTrue(any("24 hours" in message for message, _percent in updates))


# ### Cancellation ownership ###
class LunaBatchCancellationOwnershipTests(unittest.TestCase):
    def test_plan_batch_waits_for_remote_cleanup_in_owning_thread(self) -> None:
        owner = threading.get_ident()
        cancelled = threading.Event()
        check = cancelled.is_set
        cleaned = []

        def editor(_image, **kwargs):
            self.assertEqual(threading.get_ident(), owner)
            self.assertIs(kwargs["cancellation_check"], check)
            cancelled.set()
            cleaned.append(True)
            raise correction.PlanImageCorrectionCancelled(
                "Remote cancellation complete"
            )

        with (
            patch.object(
                correction,
                "_invoke_editor_cancellably",
                side_effect=AssertionError("Batch cannot detach"),
            ),
            self.assertRaises(correction.PlanImageCorrectionCancelled),
        ):
            correction._run_correction_pass(
                editor,
                np.full((32, 32, 3), 255, dtype=np.uint8),
                api_key="key",
                prompt="prompt",
                output_size=(1024, 1024),
                selected_model=LUNA_BATCH_MODEL,
                editor_was_injected=False,
                cancellation_check=check,
            )
        self.assertEqual(cleaned, [True])

    def test_reference_batch_waits_for_remote_cleanup_in_owning_thread(self) -> None:
        owner = threading.get_ident()
        cancelled = threading.Event()
        check = cancelled.is_set
        cleaned = []

        def editor(_image, **kwargs):
            self.assertEqual(threading.get_ident(), owner)
            self.assertIs(kwargs["cancellation_check"], check)
            cancelled.set()
            cleaned.append(True)
            raise correction.PlanImageCorrectionCancelled(
                "Remote cancellation complete"
            )

        with (
            patch.object(
                reference, "_create_reference_image_editor", return_value=editor
            ),
            patch.object(
                correction,
                "_invoke_editor_cancellably",
                side_effect=AssertionError("Batch cannot detach"),
            ),
            self.assertRaises(reference.ReferenceGenerationCancelled),
        ):
            reference.generate_object_reference(
                np.full((24, 32, 4), 255, dtype=np.uint8),
                "front view",
                LUNA_BATCH_MODEL,
                api_key="key",
                cancellation_check=check,
            )
        self.assertEqual(cleaned, [True])

    def test_surface_batch_keeps_cleanup_owned_but_standard_keeps_wrapper(self) -> None:
        from housemaker import surface_texture_workspace as workspace

        for model, expected_wrapper_calls in ((LUNA_BATCH_MODEL, 0), (LUNA_MODEL, 1)):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                request = workspace.SurfaceTextureRequest(
                    provider=model,
                    api_key="key",
                    reference_pngs=(_png(),),
                    reference_frame_indices=(0,),
                    surface_type="wall",
                    surface_ids=("wall",),
                    combined_area_m2=1.0,
                    prompt="wood",
                )
                worker = workspace.SurfaceTextureWorker(
                    "job", lambda _request: None, request, Path(directory)
                )
                owner = threading.get_ident()
                cleaned = []
                finished = []
                worker.cancelled.connect(
                    lambda _job, finished=finished, cleaned=cleaned: finished.append(
                        bool(cleaned)
                    )
                )

                def provider(
                    _provider,
                    _request,
                    _progress,
                    event,
                    *,
                    owner=owner,
                    worker=worker,
                    cleaned=cleaned,
                ):
                    self.assertEqual(threading.get_ident(), owner)
                    worker.cancel()
                    self.assertTrue(event.is_set())
                    cleaned.append(True)
                    raise surfaces.SurfaceTextureTaskError(
                        "Remote cancellation complete"
                    )

                with (
                    patch.object(workspace, "_invoke_provider", side_effect=provider),
                    patch.object(
                        workspace,
                        "_run_interruptible_stage",
                        side_effect=lambda operation: operation(),
                    ) as wrapper,
                ):
                    worker.run()
                self.assertEqual(wrapper.call_count, expected_wrapper_calls)
                self.assertEqual(finished, [True])


# ### Surface texture routing ###
class LunaBatchSurfaceTests(unittest.TestCase):
    def test_surface_batch_retains_selected_provider_and_real_wire_model(self) -> None:
        updates = []
        event = threading.Event()
        png = _png()

        def batch(body, **kwargs):
            self.assertEqual(body["model"], LUNA_MODEL)
            self.assertEqual(body["tools"][0]["size"], "2048x2048")
            self.assertFalse(kwargs["cancellation_check"]())
            kwargs["progress_callback"]("Queued (24-hour completion window)")
            return _result(png)

        with (
            patch.object(surfaces, "request_luna_batch_response", side_effect=batch),
            patch.object(
                surfaces,
                "_request_json",
                side_effect=AssertionError("No realtime fallback"),
            ),
        ):
            output = surfaces.request_surface_texture(
                LUNA_BATCH_MODEL,
                "key",
                (png,),
                "wood",
                progress_callback=lambda message, percent: updates.append(
                    (message, percent)
                ),
                cancel_event=event,
            )
        self.assertEqual(output.provider, LUNA_BATCH_MODEL)
        self.assertEqual(output.texture_png, png)
        self.assertTrue(any("24-hour" in message for message, _percent in updates))

    def test_surface_batch_errors_and_cancel_are_explicit(self) -> None:
        for error in (OpenAIBatchCancelled("cancelled"), OpenAIBatchError("expired")):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(
                    surfaces, "request_luna_batch_response", side_effect=error
                ),
                self.assertRaisesRegex(
                    surfaces.SurfaceTextureProviderError, str(error)
                ),
            ):
                surfaces.request_surface_texture(
                    LUNA_BATCH_MODEL, "key", (_png(),), "wood"
                )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
