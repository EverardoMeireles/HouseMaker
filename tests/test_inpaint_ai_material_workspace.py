"""Reference instructions and PBR choices belong to their individual blobs."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import test_all_reference_generation as batch_fixtures

from housemaker.generation_state import GeneratedObjectRecord
from housemaker.object_texture_variants import (
    PBR_MAP_METALLIC,
    PBR_MAP_NORMAL,
    PBR_MAP_ROUGHNESS,
    PBR_MAP_TYPES,
)
from housemaker.reference_view_presets import DEFAULT_REFERENCE_VIEW_PRESETS
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_texture_viewer import SurfaceTextureViewer


# ### Per-object UI integration without real generation ###
class InpaintAiMaterialWorkspaceTests(unittest.TestCase):
    tearDown = batch_fixtures.AllReferenceGenerationTests.tearDown
    _paint = batch_fixtures.AllReferenceGenerationTests._paint
    _start_all = batch_fixtures.AllReferenceGenerationTests._start_all
    _finish_all = batch_fixtures.AllReferenceGenerationTests._finish_all

    def setUp(self):
        batch_fixtures.AllReferenceGenerationTests.setUp(self)
        self.surfaces.surface_view = SurfaceTextureViewer(self.workspace)

    def _set_maps(self, enabled):
        for map_type, checkbox in self.workspace.pbr_map_checkboxes.items():
            checkbox.setChecked(map_type in enabled)

    def _assert_maps(self, expected):
        self.assertEqual(self.objects._get_enabled_pbr_maps(), expected)
        enabled = {map_type: map_type in expected for map_type in PBR_MAP_TYPES}
        self.assertEqual(self.objects.result_view.get_pbr_maps_enabled(), enabled)
        self.assertEqual(self.surfaces.surface_view.get_pbr_maps_enabled(), enabled)

    def _seed_preferences(self):
        prompts = ("Remove the fabric cover", "Preserve the carved details", "Show the complete object")
        maps = ((PBR_MAP_NORMAL,), (PBR_MAP_ROUGHNESS, PBR_MAP_METALLIC), ())
        for prompt, enabled in zip(prompts, maps):
            self.workspace.reference_generation_prompt_edit.setText(prompt)
            self._set_maps(enabled)
            self.workspace.next_inpaint_button.click()
        return prompts, maps

    def test_navigation_restores_each_blobs_full_prompt_and_pbr_maps(self):
        prompts, maps = self._seed_preferences()
        for prompt, enabled in zip(prompts, maps):
            self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), prompt)
            self._assert_maps(enabled)
            self.workspace.next_inpaint_button.click()

    def test_direct_blob_selection_and_seek_restore_prompt_and_material_preview(self):
        prompts, maps = self._seed_preferences()
        for frame, blob_index, index in ((0, 1, 1), (2, 0, 2), (0, 0, 0)):
            with self.subTest(frame=frame, blob_index=blob_index):
                self.workspace.seekbar.setValue(frame)
                self.workspace.video_view.select_blob(blob_index)
                self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), prompts[index])
                self._assert_maps(maps[index])

    def test_new_blob_uses_initial_defaults_not_previous_objects_preferences(self):
        self.workspace.reference_generation_prompt_edit.setText("Only for the first object")
        self._set_maps(())
        self.workspace.next_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt)
        self._assert_maps(PBR_MAP_TYPES)
        self.workspace.next_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt)
        self._assert_maps(PBR_MAP_TYPES)

    def test_empty_frame_does_not_promote_previous_blobs_prompt_to_new_defaults(self):
        self.workspace.reference_generation_prompt_edit.setText("Only for the first object")
        self.workspace.seekbar.setValue(1)
        self.assertIsNone(self.objects.current_inpaint_reference_key())
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt)
        self.workspace.reference_generation_strength_spinbox.setValue(0.60)
        self.workspace.seekbar.setValue(0)
        self.workspace.video_view.select_blob(1)
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt)

    def test_programmatic_material_edit_refreshes_both_previews_without_affecting_other_blob(self):
        key = self.objects.current_inpaint_reference_key()
        self.assertTrue(self.objects.set_inpaint_pbr_maps(key, (PBR_MAP_NORMAL,)))
        self._assert_maps((PBR_MAP_NORMAL,))
        self.workspace.next_inpaint_button.click()
        self._assert_maps(PBR_MAP_TYPES)
        self.workspace.previous_inpaint_button.click()
        self._assert_maps((PBR_MAP_NORMAL,))

    def test_cleared_prompt_stays_empty_after_navigation_without_leaking_defaults(self):
        self.workspace.reference_generation_prompt_edit.setText("Remove clutter")
        self.workspace.next_inpaint_button.click()
        self.workspace.reference_generation_prompt_edit.setText("Repair the legs")
        self.workspace.previous_inpaint_button.click()
        self.workspace.reference_generation_prompt_edit.clear()
        self.workspace.next_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), "Repair the legs")
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), "")

    def test_single_reference_uses_exact_selected_blobs_visible_prompt(self):
        prompts, _maps = self._seed_preferences()
        self.workspace.next_inpaint_button.click()
        self.workspace.generate_reference_button.click()
        worker = batch_fixtures._ControlledReferenceThread.created[-1]
        self.assertEqual(worker.prompt, prompts[1])
        worker.complete(batch_fixtures._reference(141))
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), prompts[0])

    def test_reference_batch_freezes_independent_full_prompts_without_navigating(self):
        prompts, _maps = self._seed_preferences()
        key = self.objects.current_inpaint_reference_key()
        workers = self._start_all()
        expected = list(prompts)
        self.assertEqual([worker.prompt for worker in workers], expected)
        self.assertEqual(self.objects.current_inpaint_reference_key(), key)
        self.workspace.reference_generation_prompt_edit.setText("For a later request")
        self.assertEqual([worker.prompt for worker in workers], expected)
        self._finish_all(workers)

    def test_full_generation_snapshots_each_blobs_materials_independently(self):
        _prompts, maps = self._seed_preferences()
        self.objects.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test"))
        self.workspace.next_inpaint_button.click()
        selected = self.objects.current_inpaint_reference_key()
        requests = self.objects._build_generation_requests()
        self.assertEqual([request.enabled_pbr_maps for request in requests], list(maps))
        self.assertEqual(self.objects.current_inpaint_reference_key(), selected)
        self._set_maps(PBR_MAP_TYPES)
        self.assertEqual([request.enabled_pbr_maps for request in requests], list(maps))

    def test_geometry_then_texture_batch_uses_each_blobs_materials(self):
        _prompts, maps = self._seed_preferences()
        self.objects.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test"))
        with patch.object(self.objects, "_start_generation", side_effect=["a", "b", "c"]):
            self.objects.generate_geometry()
        self.objects._data.generated_objects = [
            GeneratedObjectRecord(
                object_id=object_id, frame_index=frame_index, object_name=object_id,
                pipeline={}, provider_task_id=object_id, asset_path=f"{object_id}.glb",
            )
            for object_id, frame_index in (("a", 0), ("b", 0), ("c", 2))
        ]
        with patch.object(
            self.objects, "_build_texture_regeneration_request", side_effect=[Mock(), Mock(), Mock()],
        ) as build, patch.object(self.objects, "_start_texture_regeneration", return_value=True):
            self.assertTrue(self.objects.generate_selected_object_texture())
        self.assertEqual(
            [call.kwargs["enabled_pbr_maps_override"] for call in build.call_args_list],
            list(maps),
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
