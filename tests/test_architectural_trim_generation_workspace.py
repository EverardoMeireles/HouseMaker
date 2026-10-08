# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtWidgets import QApplication

from housemaker.generation_workspace import (
    OBJECT_OPERATION_GENERATE_MODEL,
    GenerationWorkspace,
    _ActiveObjectOperation,
    _materialize_texture_regeneration_preflight,
)
from housemaker.glb import GeneratedModel, import_generated_glb

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _trim_component_model() -> GeneratedModel:
    mesh = trimesh.creation.box(extents=(1.0, 0.1, 0.2))
    glb_bytes = bytes(trimesh.Scene(mesh).export(file_type="glb"))
    return import_generated_glb(glb_bytes)


# ### Architectural-trim generation routing tests ###
class ArchitecturalTrimGenerationWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = GenerationWorkspace(
            asset_directory=Path(self._temporary_directory.name) / "generated"
        )
        self.trim_id = "a" * 32
        self.object_id = "architectural-trim-test-object"
        self.record = self.workspace.register_architectural_trim_component_model(
            level_index=2,
            trim_id=self.trim_id,
            object_id=self.object_id,
            object_name="Cornice",
            model=_trim_component_model(),
        )
        self.assertTrue(
            self.workspace.set_architectural_trim_editing_target(
                level_index=2,
                trim_id=self.trim_id,
                object_id=self.object_id,
                object_name="Cornice",
            )
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def test_generate_geometry_replaces_selected_trim_component(self) -> None:
        generation_request = object()
        with (
            patch.object(
                self.workspace,
                "_build_generation_request",
                return_value=generation_request,
            ) as build_request,
            patch.object(
                self.workspace,
                "_start_generation",
                return_value="trim-generation-operation",
            ) as start_generation,
        ):
            self.workspace.generate_geometry()

        build_request.assert_called_once_with(geometry_only=True)
        start_generation.assert_called_once()
        self.assertIs(start_generation.call_args.args[0], generation_request)
        target = start_generation.call_args.kwargs["architectural_trim_target"]
        self.assertEqual(target.level_index, 2)
        self.assertEqual(target.trim_id, self.trim_id)
        self.assertEqual(target.object_id, self.object_id)
        self.assertIs(
            start_generation.call_args.kwargs["replaced_object_record"],
            self.record,
        )
        self.assertNotIn("door_slot_target", start_generation.call_args.kwargs)

    def test_generate_texture_uses_selected_trim_instead_of_other_object(
        self,
    ) -> None:
        texture_request = object()
        selected_ids: list[str | None] = []

        def build_request() -> object:
            selected_ids.append(self.workspace._selected_object_id)
            return texture_request

        with (
            patch.object(
                self.workspace,
                "_build_texture_regeneration_request",
                side_effect=build_request,
            ) as build_texture_request,
            patch.object(
                self.workspace,
                "_start_texture_regeneration",
                return_value=True,
            ) as start_texture_regeneration,
        ):
            self.assertTrue(
                self.workspace.generate_selected_object_texture()
            )

        build_texture_request.assert_called_once_with()
        start_texture_regeneration.assert_called_once_with(texture_request)
        self.assertEqual(selected_ids, [self.object_id])
        target = self.workspace._architectural_trim_editing_target
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.object_id, self.object_id)

    def test_uvless_trim_seed_can_prepare_texture_regeneration(self) -> None:
        reference_bgra = np.zeros((16, 16, 4), dtype=np.uint8)
        reference_bgra[:, :, 3] = 255
        with (
            patch.object(
                self.workspace,
                "_can_regenerate_object_texture",
                return_value=True,
            ),
            patch.object(
                self.workspace,
                "_build_current_object_generation_reference",
                return_value=reference_bgra,
            ),
        ):
            preflight = self.workspace._build_texture_regeneration_request()

        self.assertIsNotNone(preflight)
        assert preflight is not None
        materialized = _materialize_texture_regeneration_preflight(
            preflight,
            self.workspace._asset_directory,
        )

        self.assertEqual(materialized.request.object_id, self.object_id)
        self.assertTrue(materialized.request.model_glb)

    def test_older_trim_generation_completion_keeps_newer_trim_selected(
        self,
    ) -> None:
        first_target = self.workspace._architectural_trim_editing_target
        self.assertIsNotNone(first_target)
        assert first_target is not None
        operation = _ActiveObjectOperation(
            kind=OBJECT_OPERATION_GENERATE_MODEL,
            target_object_id=self.object_id,
            architectural_trim_target=first_target,
            replaced_object_record=self.record,
        )
        self.workspace._active_object_operation = operation
        self.addCleanup(
            setattr,
            self.workspace,
            "_active_object_operation",
            None,
        )

        second_trim_id = "b" * 32
        second_object_id = "architectural-trim-second-object"
        second_model = _trim_component_model()
        second_record = self.workspace.register_architectural_trim_component_model(
            level_index=2,
            trim_id=second_trim_id,
            object_id=second_object_id,
            object_name="Second cornice",
            model=second_model,
        )
        self.assertTrue(
            self.workspace.set_architectural_trim_editing_target(
                level_index=2,
                trim_id=second_trim_id,
                object_id=second_object_id,
                object_name=second_record.object_name,
            )
        )

        replacement_model = _trim_component_model()
        replacement_asset_path = self.workspace._persist_meshy_named_asset(
            "first-cornice-replacement.glb",
            replacement_model.glb_bytes,
        )
        replacement_record = replace(
            self.record,
            asset_path=replacement_asset_path,
        )

        self.assertTrue(
            self.workspace._commit_generated_model_record(
                replacement_record,
                replacement_model,
                self.record,
                operation_id=None,
            )
        )

        active_target = self.workspace._architectural_trim_editing_target
        self.assertIsNotNone(active_target)
        assert active_target is not None
        self.assertEqual(active_target.object_id, second_object_id)
        self.assertEqual(self.workspace._selected_object_id, second_object_id)
        self.assertIs(self.workspace.result_view.model, second_model)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
