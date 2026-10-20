"""Resident backends must preserve the normal reference preparation contract."""

# ### Imports ###
from __future__ import annotations

import threading
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from housemaker.plan_correction_models import PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1
from housemaker.reference_generation import (
    ReferenceGenerationError,
    open_object_reference_session,
)
from housemaker.reference_generation_models import (
    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
)


# ### Image fixtures ###
def _output_png() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (1024, 1024), "white").save(stream, format="PNG")
    return stream.getvalue()


# ### Session orchestration tests ###
class ObjectReferenceSessionTests(unittest.TestCase):
    def _exercise_session(self, model, backend_class):
        entered_thread_ids = []
        output = _output_png()

        def generate(*args, **kwargs):
            entered_thread_ids.append(threading.get_ident())
            with Image.open(BytesIO(args[0])) as prepared:
                self.assertEqual(prepared.size, (1024, 1024))
                self.assertEqual(prepared.mode, "RGB")
            self.assertIn("clean white background", kwargs["prompt"])
            return output

        backend = Mock(side_effect=generate)
        backend.__enter__ = Mock(return_value=backend)
        backend.__exit__ = Mock(return_value=False)
        source = np.full((8, 12, 4), 255, dtype=np.uint8)
        with (
            patch(f"housemaker.local_reference_image_session.{backend_class}", return_value=backend) as factory,
            patch("housemaker.reference_generation._create_reference_image_editor") as one_shot,
            patch("housemaker.plan_image_correction._invoke_editor_cancellably") as detached_transport,
        ):
            with open_object_reference_session(model) as session:
                factory.assert_called_once()
                backend.assert_not_called()
                for _ in range(2):
                    image = session(source, "Front view", cancellation_check=lambda: False)
                    self.assertEqual(image.shape, (1024, 1024, 4))
            self.assertEqual(backend.call_count, 2)
            backend.__enter__.assert_called_once()
            backend.__exit__.assert_called_once()
            one_shot.assert_not_called()
            detached_transport.assert_not_called()
        self.assertEqual(entered_thread_ids, [threading.get_ident()] * 2)
        return backend

    def test_qwen_reuses_backend_without_detaching_native_requests(self):
        self._exercise_session(PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1, "QwenReferenceImageSession")

    def test_sdxl_reuses_backend_with_same_default_generation_parameters(self):
        backend = self._exercise_session(
            REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION, "StableDiffusionReferenceSession",
        )
        self.assertEqual(backend.call_args.kwargs["strength"], 0.45)
        self.assertIn("negative_prompt", backend.call_args.kwargs)

    def test_remote_provider_cannot_start_a_local_session(self):
        with (
            self.assertRaisesRegex(ReferenceGenerationError, "local image model"),
            open_object_reference_session("gpt-image-2"),
        ):
            self.fail("The remote provider must not open a local session")


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
