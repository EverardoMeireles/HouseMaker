# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import trimesh
from PIL import Image
from PySide6.QtWidgets import QApplication
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from housemaker.generation_state import GeneratedObjectRecord, GenerationData
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.main import BlueprintWorkspace
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _build_textured_glb() -> bytes:
    rgba = np.empty((512, 512, 4), dtype=np.uint8)
    rgba[:, :] = (90, 120, 150, 255)
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    mesh.visual = TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2), dtype=float),
        material=PBRMaterial(
            name="Preview source",
            baseColorTexture=Image.fromarray(rgba, mode="RGBA"),
        ),
    )
    return bytes(trimesh.Scene(mesh).export(file_type="glb"))


def _generated_record(asset_path: str) -> GeneratedObjectRecord:
    return GeneratedObjectRecord(
        object_id="generated-chair",
        frame_index=0,
        object_name="Generated chair",
        pipeline={},
        provider="meshy",
        provider_task_id="task-generated-chair",
        asset_path=asset_path,
    )


# ### Generated-object preview loading tests ###
class GeneratedObjectPreviewLoadingTests(unittest.TestCase):
    def test_project_restore_defers_selected_glb_and_uses_preview_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            asset_directory = Path(temporary_directory)
            payload = _build_textured_glb()
            (asset_directory / "chair.glb").write_bytes(payload)
            workspace = GenerationWorkspace(asset_directory=asset_directory)
            try:
                with patch(
                    "housemaker.generation_workspace.import_generated_glb"
                ) as authoritative_import:
                    workspace.set_data(
                        GenerationData(
                            generated_objects=[_generated_record("chair.glb")]
                        ),
                        defer_file_backed_preview=True,
                    )
                    authoritative_import.assert_not_called()

                    proxy = np.empty((128, 256, 4), dtype=np.uint8)
                    proxy[:, :] = (15, 25, 35, 255)
                    preview = workspace.get_generated_object_preview_model(
                        "generated-chair",
                        {"base_color": proxy},
                    )

                    authoritative_import.assert_not_called()
                self.assertEqual(workspace._selected_object_id, "generated-chair")
                self.assertIsNone(workspace.result_view.model)
                self.assertIsNotNone(preview)
                assert preview is not None
                self.assertTrue(preview.preview_only)
                self.assertEqual(preview.glb_bytes, payload)
                self.assertNotIn(
                    "generated-chair",
                    workspace._generated_model_cache,
                )
                self.assertIs(
                    workspace.get_generated_object_preview_model(
                        "generated-chair",
                        {"base_color": proxy.copy()},
                    ),
                    preview,
                )
            finally:
                workspace.shutdown()
                workspace.close()
                workspace.deleteLater()
                _qt_application.processEvents()

    def test_canvas_footprints_skip_door_and_trim_model_work(self) -> None:
        levels = create_default_levels()
        current_level = levels[GROUND_LEVEL_INDEX]
        build_models = Mock(return_value=())
        workspace = SimpleNamespace(
            canvas=Mock(),
            levels=levels,
            current_level_index=GROUND_LEVEL_INDEX,
            current_level=current_level,
            generation=Mock(),
            _build_placed_generated_models=build_models,
            _sync_blueprint_placed_object_selection=Mock(),
        )

        BlueprintWorkspace._sync_canvas_placed_object_profiles(workspace)

        build_models.assert_called_once_with(
            include_excluded_levels=True,
            include_instances=True,
            include_door_placements=False,
            preview_only=True,
            precomputed_architectural_trim_models=(),
        )
        workspace.canvas.set_placed_object_profiles.assert_called_once_with([])

    def test_scene_model_selector_separates_preview_and_authoritative_loads(
        self,
    ) -> None:
        authoritative = object()
        preview = object()
        proxy_maps = {"base_color": np.zeros((1, 1, 4), dtype=np.uint8)}
        source = object()
        generation = SimpleNamespace(
            get_generated_object_model=Mock(return_value=authoritative),
            get_active_texture_variant=Mock(return_value=object()),
            get_generated_object_preview_model=Mock(return_value=preview),
        )
        workspace = SimpleNamespace(
            generation=generation,
            _build_atlas_object_texture_source=Mock(return_value=source),
            _atlas_source_preview_maps=Mock(return_value=proxy_maps),
        )

        self.assertIs(
            BlueprintWorkspace._get_scene_generated_object_model(
                workspace,
                "generated-chair",
                symmetry=None,
                preview_only=False,
            ),
            authoritative,
        )
        self.assertIs(
            BlueprintWorkspace._get_scene_generated_object_model(
                workspace,
                "generated-chair",
                symmetry=None,
                preview_only=True,
            ),
            preview,
        )
        generation.get_generated_object_model.assert_called_once_with(
            "generated-chair"
        )
        generation.get_generated_object_preview_model.assert_called_once_with(
            "generated-chair",
            proxy_maps,
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
