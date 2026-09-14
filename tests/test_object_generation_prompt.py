# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from housemaker.generation_workspace import (
    GenerationRequest,
    GenerationWorkspace,
    MeshyImagePlanner,
)
from housemaker.meshy_generation import (
    MAX_IMAGE_TO_3D_TEXTURE_PROMPT_CHARACTERS,
    MeshyGenerationResult,
)
from housemaker.settings_widget import GenerationServiceSettings

# ### Test application ###
_qt_application = QApplication.instance() or QApplication([])


# ### Request tests ###
class ObjectGenerationPromptRequestTests(unittest.TestCase):
    def test_generation_request_normalizes_optional_ai_prompt(self) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.full((4, 4, 4), 255, dtype=np.uint8),
            settings=GenerationServiceSettings(),
            ai_prompt="  aged oak with brass hardware  ",
        )

        self.assertEqual(request.ai_prompt, "aged oak with brass hardware")

    def test_generation_request_rejects_oversized_ai_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 800"):
            GenerationRequest(
                frame_index=0,
                selected_object_bgra=np.full(
                    (4, 4, 4),
                    255,
                    dtype=np.uint8,
                ),
                settings=GenerationServiceSettings(),
                ai_prompt=("x" * (MAX_IMAGE_TO_3D_TEXTURE_PROMPT_CHARACTERS + 1)),
            )


# ### Planner tests ###
class ObjectGenerationPromptPlannerTests(unittest.TestCase):
    def test_direct_textured_generation_forwards_ai_prompt(self) -> None:
        request = GenerationRequest(
            frame_index=0,
            selected_object_bgra=np.full((4, 4, 4), 255, dtype=np.uint8),
            settings=GenerationServiceSettings(meshy_api_key="test-key"),
            ai_prompt="painted blue wood",
        )
        provider_result = MeshyGenerationResult(
            task_id="task",
            glb_bytes=b"provider glb",
        )

        with (
            patch(
                "housemaker.generation_workspace.request_image_to_3d_model",
                return_value=provider_result,
            ) as request_model,
            patch(
                "housemaker.generation_workspace."
                "_remove_safe_duplicates_from_meshy_result",
                return_value=(provider_result, object()),
            ),
            patch(
                "housemaker.generation_workspace._scan_project_provider_result",
                return_value=provider_result,
            ),
            patch(
                "housemaker.generation_workspace.request_retextured_model"
            ) as request_retexture,
        ):
            result = MeshyImagePlanner().plan(request)

        self.assertIs(result, provider_result)
        self.assertEqual(
            request_model.call_args.kwargs["texture_prompt"],
            "painted blue wood",
        )
        request_retexture.assert_not_called()


# ### Workspace tests ###
class ObjectGenerationPromptWorkspaceTests(unittest.TestCase):
    def test_workspace_exposes_bounded_standalone_ai_prompt_editor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = GenerationWorkspace(asset_directory=Path(temporary_directory))
            try:
                self.assertEqual(
                    workspace.ai_prompt_edit.objectName(),
                    "object_generation_ai_prompt_edit",
                )
                self.assertEqual(
                    workspace.ai_prompt_edit.maxLength(),
                    MAX_IMAGE_TO_3D_TEXTURE_PROMPT_CHARACTERS,
                )
            finally:
                workspace.shutdown()


if __name__ == "__main__":
    unittest.main()
