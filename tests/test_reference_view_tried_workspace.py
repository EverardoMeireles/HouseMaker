"""Successful reference generations mark tried views without AI view selection."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import test_all_reference_generation as batch_fixtures
import test_local_reference_batch_workspace as local_fixtures
import test_reference_generation_workspace as reference_fixtures

from housemaker.reference_view_presets import DEFAULT_REFERENCE_VIEW_PRESETS


# ### Shared view-label assertions ###
def _label(workspace, preset_id: str) -> str:
    combo = workspace.reference_generation_view_combo
    return combo.itemText(combo.findData(preset_id))


def _tried_label(preset) -> str:
    return f"{preset.name} (Already tried)"


# ### Single-reference and per-object batch labels ###
class ReferenceViewTriedWorkspaceTests(unittest.TestCase):
    setUp = batch_fixtures.AllReferenceGenerationTests.setUp
    tearDown = batch_fixtures.AllReferenceGenerationTests.tearDown
    _paint = batch_fixtures.AllReferenceGenerationTests._paint
    _start_all = batch_fixtures.AllReferenceGenerationTests._start_all

    def _start_single(self):
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        self.workspace.generate_reference_button.click()
        return batch_fixtures._ControlledReferenceThread.created[-1]

    def test_success_marks_only_used_view_and_keeps_selection_and_item_data(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        key = self.objects.current_inpaint_reference_key()
        worker = self._start_single()
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)
        worker.complete(batch_fixtures._reference(123))
        combo = self.workspace.reference_generation_view_combo
        self.assertEqual(combo.currentData(), preset.preset_id)
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        self.assertEqual(self.objects.get_inpaint_tried_reference_views(key),
                         frozenset({(preset.preset_id, preset.prompt)}))
        for other in DEFAULT_REFERENCE_VIEW_PRESETS[1:]:
            self.assertEqual(_label(self.workspace, other.preset_id), other.name)
        self.assertEqual(len(self.manager.jobs()), 1)
        self.assertEqual(self.manager.jobs()[0].status, "completed")

    def test_retry_does_not_duplicate_suffix_and_original_prompt_is_unchanged(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        for value in (123, 124):
            worker = self._start_single()
            self.assertEqual(worker.prompt, preset.prompt)
            worker.complete(batch_fixtures._reference(value))
            self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        self.assertEqual(self.workspace._reference_view_presets[0], preset)

    def test_two_tried_views_accumulate_for_one_blob(self):
        for preset, value in zip(DEFAULT_REFERENCE_VIEW_PRESETS[:2], (123, 124)):
            combo = self.workspace.reference_generation_view_combo
            combo.setCurrentIndex(combo.findData(preset.preset_id))
            self._start_single().complete(batch_fixtures._reference(value))
        for preset in DEFAULT_REFERENCE_VIEW_PRESETS[:2]:
            self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))

    def test_editing_visible_prompt_does_not_make_a_tried_view_new_again(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        self.workspace.reference_generation_prompt_edit.setText(f"{preset.prompt}, Keep the original colors")
        self._start_single().complete(batch_fixtures._reference(123))
        self.workspace.reference_generation_prompt_edit.setText(f"{preset.prompt}, Remove the fabric cover")
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        self.assertEqual(self.objects.get_inpaint_tried_reference_views(
            self.objects.current_inpaint_reference_key(),
        ), frozenset({(preset.preset_id, preset.prompt)}))

    def test_labels_follow_blob_navigation_not_the_previously_selected_blob(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        self._start_single().complete(batch_fixtures._reference(123))
        self.workspace.next_inpaint_button.click()
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))

    def test_failed_generation_does_not_mark_view(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        self._start_single().complete(error="The image provider failed")
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)

    def test_cancelled_generation_does_not_mark_view(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        worker = self._start_single()
        self.workspace.generate_reference_button.click()
        worker.complete(batch_fixtures._reference(123))
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)
        self.assertEqual(self.manager.jobs()[0].status, "cancelled")

    def test_stale_result_after_navigation_does_not_mark_either_blob(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        key = self.objects.current_inpaint_reference_key()
        worker = self._start_single()
        self.workspace.next_inpaint_button.click()
        worker.complete(batch_fixtures._reference(123))
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)
        self.assertFalse(self.objects.get_inpaint_tried_reference_views(key))
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)

    def test_accepted_image_is_tried_even_if_symmetry_inference_fails(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        worker = self._start_single()
        worker.symmetry_error = "The symmetry model is unavailable"
        worker.complete(batch_fixtures._reference(123))
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        self.assertEqual(self.manager.jobs()[0].status, "failed")

    def test_obsolete_inferred_view_result_cannot_change_selection_or_job_status(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        worker = self._start_single()
        self.assertFalse(hasattr(worker, "reference_views"))
        worker.view_result = SimpleNamespace(preset_id="side", name="Side/profile")
        worker.view_error = "A removed view-inference stage must not be consulted"
        worker.complete(batch_fixtures._reference(123))
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), preset.preset_id)
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        self.assertEqual(self.manager.jobs()[0].status, "completed")

    def test_prompt_edit_invalidates_label_but_rename_with_same_prompt_preserves_it(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        self._start_single().complete(batch_fixtures._reference(123))
        renamed = replace(preset, name="Renamed view")
        self.workspace.set_reference_view_presets((renamed, *DEFAULT_REFERENCE_VIEW_PRESETS[1:]))
        self.assertEqual(_label(self.workspace, renamed.preset_id), _tried_label(renamed))
        edited = replace(renamed, prompt="A different camera angle")
        self.workspace.set_reference_view_presets((edited, *DEFAULT_REFERENCE_VIEW_PRESETS[1:]))
        self.assertEqual(_label(self.workspace, edited.preset_id), edited.name)
        self._start_single().complete(batch_fixtures._reference(124))
        self.assertEqual(_label(self.workspace, edited.preset_id), _tried_label(edited))

    def test_completion_marks_captured_prompt_not_a_preset_edited_during_generation(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        worker = self._start_single()
        edited = replace(preset, prompt="A prompt edited while rendering")
        self.workspace.set_reference_view_presets((edited, *DEFAULT_REFERENCE_VIEW_PRESETS[1:]))
        worker.complete(batch_fixtures._reference(123))
        self.assertEqual(_label(self.workspace, edited.preset_id), edited.name)
        self.workspace.set_reference_view_presets(DEFAULT_REFERENCE_VIEW_PRESETS)
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))

    def test_custom_prompt_does_not_decorate_write_prompt_item(self):
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(""))
        self.workspace.reference_generation_prompt_edit.setText("Recreate this object clearly")
        self._start_single().complete(batch_fixtures._reference(123))
        self.assertEqual(_label(self.workspace, ""), "(Write prompt)")
        self.assertTrue(all(_label(self.workspace, preset.preset_id) == preset.name
                            for preset in DEFAULT_REFERENCE_VIEW_PRESETS))

    def test_batch_marks_each_blob_only_when_its_own_image_is_accepted(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        keys = self.objects._current_inpaint_keys()
        workers = self._start_all()
        self.workspace.next_inpaint_button.click()
        workers[0].complete(batch_fixtures._reference(101))
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)
        workers[1].complete(batch_fixtures._reference(102))
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        workers[2].complete(error="This object's render failed")
        self.assertEqual([bool(self.objects.get_inpaint_tried_reference_views(key)) for key in keys],
                         [True, True, False])
        self.workspace.next_inpaint_button.click()
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)


# ### Grouped local model phases ###
class LocalReferenceViewTriedWorkspaceTests(unittest.TestCase):
    setUp = local_fixtures.LocalReferenceBatchWorkspaceTests.setUp
    tearDown = local_fixtures.LocalReferenceBatchWorkspaceTests.tearDown
    _paint = local_fixtures.LocalReferenceBatchWorkspaceTests._paint
    _start_local = local_fixtures.LocalReferenceBatchWorkspaceTests._start_local

    def test_local_image_acceptance_marks_view_before_later_symmetry_phase(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        worker = self._start_local()
        self.assertFalse(hasattr(worker, "reference_views"))
        worker.render(0, 101)
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        self.workspace.next_inpaint_button.click()
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)
        worker.render(1, 102)
        worker.render(2, 103)
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        worker.decide(0, False, error="Symmetry inference failed")
        worker.decide(1, True)
        worker.decide(2, False)
        worker.complete()
        for _ in range(3):
            self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
            self.workspace.next_inpaint_button.click()

    def test_local_stale_image_does_not_mark_replaced_input(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        worker = self._start_local()
        self.workspace._activate_temporary_object_reference(
            batch_fixtures._reference(77), source_label="New pasted reference",
        )
        worker.render(0, 101)
        for index in (1, 2):
            worker.render(index, 101 + index)
            worker.decide(index, False)
        worker.complete()
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)


# ### Standalone external references and settings isolation ###
class ExternalReferenceViewTriedWorkspaceTests(unittest.TestCase):
    setUp = reference_fixtures.ReferenceGenerationWorkspaceTests.setUp
    tearDown = reference_fixtures.ReferenceGenerationWorkspaceTests.tearDown
    _apply_reference_generation_model = (
        reference_fixtures.ReferenceGenerationWorkspaceTests._apply_reference_generation_model
    )

    def _generate_external(self, value: int):
        self.workspace.generate_reference_button.click()
        worker = reference_fixtures._ControlledReferenceThread.created[-1]
        worker.complete(reference_fixtures._opaque_reference(value))

    def test_external_retry_keeps_marks_but_loading_another_image_clears_them(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        self.workspace._activate_temporary_object_reference(
            reference_fixtures._opaque_reference(70), source_label="Loaded image",
        )
        self._generate_external(123)
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        self._generate_external(124)
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        self.workspace._activate_temporary_object_reference(
            reference_fixtures._opaque_reference(71), source_label="Pasted image",
        )
        self.assertEqual(_label(self.workspace, preset.preset_id), preset.name)

    def test_tried_suffix_never_changes_saved_preset_names_or_settings_list(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        self.workspace._activate_temporary_object_reference(
            reference_fixtures._opaque_reference(), source_label="Loaded image",
        )
        self._generate_external(123)
        self.assertEqual(_label(self.workspace, preset.preset_id), _tried_label(preset))
        self.assertEqual(self.workspace._reference_view_presets, DEFAULT_REFERENCE_VIEW_PRESETS)
        items = self.settings_widget.reference_view_preset_list
        self.assertEqual([items.item(index).text() for index in range(items.count())],
                         [preset.name for preset in DEFAULT_REFERENCE_VIEW_PRESETS])


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
