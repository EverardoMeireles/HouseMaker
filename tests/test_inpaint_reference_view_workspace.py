"""Manual reference views belong to each blob in single and batch generation."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import test_all_reference_generation as batch_fixtures
import test_local_reference_batch_worker as worker_fixtures
import test_local_reference_batch_workspace as local_fixtures

from housemaker.reference_view_presets import DEFAULT_REFERENCE_VIEW_PRESETS


# ### View-selection helpers ###
def _select(workspace, preset_id: str) -> None:
    combo = workspace.reference_generation_view_combo
    combo.setCurrentIndex(combo.findData(preset_id))


def _prompt(view_prompt: str, extra: str = "") -> str:
    return ", ".join(part for part in (view_prompt, extra) if part)


def _label(workspace, preset_id: str) -> str:
    combo = workspace.reference_generation_view_combo
    return combo.itemText(combo.findData(preset_id))


# ### Navigation and API request integration ###
class InpaintReferenceViewWorkspaceTests(unittest.TestCase):
    setUp = batch_fixtures.AllReferenceGenerationTests.setUp
    tearDown = batch_fixtures.AllReferenceGenerationTests.tearDown
    _paint = batch_fixtures.AllReferenceGenerationTests._paint
    _start_all = batch_fixtures.AllReferenceGenerationTests._start_all

    def _start_single(self):
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        self.workspace.generate_reference_button.click()
        return batch_fixtures._ControlledReferenceThread.created[-1]

    def _set_three_views(self):
        presets = DEFAULT_REFERENCE_VIEW_PRESETS[1:4]
        for preset in presets:
            _select(self.workspace, preset.preset_id)
            self.workspace.next_inpaint_button.click()
        return presets

    def test_next_and_previous_restore_independent_views_including_other_frames(self):
        presets = self._set_three_views()
        keys = self.objects._current_inpaint_keys()
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), presets[0].preset_id)
        self.workspace.next_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), presets[1].preset_id)
        self.workspace.next_inpaint_button.click()
        self.assertEqual(self.objects._displayed_frame_index, 2)
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), presets[2].preset_id)
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), presets[1].preset_id)
        for key, preset in zip(keys, presets):
            self.assertEqual(self.objects.get_inpaint_reference_view_preferences(key)[0], preset.preset_id)

    def test_selecting_blob_directly_and_seeking_restore_its_own_view(self):
        presets = self._set_three_views()
        self.workspace.video_view.select_blob(1)
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), presets[1].preset_id)
        self.workspace.seekbar.setValue(2)
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), presets[2].preset_id)
        self.workspace.seekbar.setValue(0)
        self.workspace.video_view.select_blob(1)
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), presets[1].preset_id)

    def test_unconfigured_blob_uses_initial_default_not_previous_objects_selection(self):
        _select(self.workspace, DEFAULT_REFERENCE_VIEW_PRESETS[3].preset_id)
        self.workspace.next_inpaint_button.click()
        self.assertEqual(
            self.workspace.reference_generation_view_combo.currentData(),
            DEFAULT_REFERENCE_VIEW_PRESETS[0].preset_id,
        )
        self.workspace.next_inpaint_button.click()
        self.assertEqual(
            self.workspace.reference_generation_view_combo.currentData(),
            DEFAULT_REFERENCE_VIEW_PRESETS[0].preset_id,
        )

    def test_custom_prompts_are_restored_per_blob_without_leaking_to_another_blob(self):
        _select(self.workspace, "")
        self.workspace.reference_generation_prompt_edit.setText("A clear front view of the first object")
        self.workspace.next_inpaint_button.click()
        _select(self.workspace, "")
        self.workspace.reference_generation_prompt_edit.setText("Show the second object from above")
        self.workspace.next_inpaint_button.click()
        _select(self.workspace, DEFAULT_REFERENCE_VIEW_PRESETS[3].preset_id)
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), "")
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), "Show the second object from above")
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), "")
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), "A clear front view of the first object")

    def test_write_prompt_retains_current_preset_text_without_resurrecting_an_old_draft(self):
        _select(self.workspace, "")
        self.workspace.reference_generation_prompt_edit.setText("Keep this draft for the first object")
        _select(self.workspace, DEFAULT_REFERENCE_VIEW_PRESETS[3].preset_id)
        self.workspace.next_inpaint_button.click()
        _select(self.workspace, "")
        self.workspace.reference_generation_prompt_edit.setText("A separate second-object draft")
        self.workspace.previous_inpaint_button.click()
        _select(self.workspace, "")
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(),
            DEFAULT_REFERENCE_VIEW_PRESETS[3].prompt,
        )

    def test_edited_preset_prompt_is_individual_and_preserves_the_view_selection(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        edited = _prompt(preset.prompt, "Remove clutter and preserve its colors")
        self.workspace.reference_generation_prompt_edit.setText(edited)
        for expected in (edited, preset.prompt, preset.prompt):
            self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), preset.preset_id)
            self.assertEqual(
                self.workspace.reference_generation_prompt_edit.text(),
                expected,
            )
            self.workspace.next_inpaint_button.click()

    def test_single_reference_uses_current_blobs_view_and_marks_only_that_view(self):
        presets = self._set_three_views()
        self.workspace.next_inpaint_button.click()
        key = self.objects.current_inpaint_reference_key()
        self.workspace.reference_generation_prompt_edit.setText(_prompt(presets[1].prompt, "Keep the original material"))
        worker = self._start_single()
        self.assertEqual(worker.prompt, _prompt(presets[1].prompt, "Keep the original material"))
        worker.complete(batch_fixtures._reference(123))
        self.assertEqual(
            self.objects.get_inpaint_tried_reference_views(key),
            frozenset({(presets[1].preset_id, presets[1].prompt)}),
        )
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), presets[1].preset_id)
        self.assertIn("(Already tried)", _label(self.workspace, presets[1].preset_id))
        self.assertEqual(len(batch_fixtures._ControlledReferenceThread.created), 1)

    def test_api_batch_freezes_each_objects_full_edited_view_prompt(self):
        presets = self._set_three_views()
        extras = ("Keep the original material", "Remove the fabric cover", "")
        for preset, extra in zip(presets, extras):
            self.workspace.reference_generation_prompt_edit.setText(_prompt(preset.prompt, extra))
            self.workspace.next_inpaint_button.click()
        workers = self._start_all()
        expected = [_prompt(preset.prompt, extra) for preset, extra in zip(presets, extras)]
        self.assertEqual([worker.prompt for worker in workers], expected)
        self.workspace.reference_generation_prompt_edit.setText("A later change")
        _select(self.workspace, DEFAULT_REFERENCE_VIEW_PRESETS[0].preset_id)
        for worker, value in zip(workers, (101, 102, 103)):
            worker.complete(batch_fixtures._reference(value))
        self.assertEqual([worker.prompt for worker in workers], expected)
        keys = self.objects._current_inpaint_keys()
        for key, preset in zip(keys, presets):
            self.assertEqual(
                self.objects.get_inpaint_tried_reference_views(key),
                frozenset({(preset.preset_id, preset.prompt)}),
            )
        self.assertEqual(len(batch_fixtures._ControlledReferenceThread.created), 3)

    def test_api_batch_uses_each_custom_prompt_and_default_for_unconfigured_blob(self):
        _select(self.workspace, "")
        self.workspace.reference_generation_prompt_edit.setText("Show the first object from the front")
        self.workspace.next_inpaint_button.click()
        _select(self.workspace, "")
        self.workspace.reference_generation_prompt_edit.setText("Show the second object from the side")
        self.workspace.previous_inpaint_button.click()
        self.workspace.reference_generation_prompt_edit.setText("Show the first object from the front, White background")
        workers = self._start_all()
        self.assertEqual([worker.prompt for worker in workers], [
            "Show the first object from the front, White background",
            "Show the second object from the side",
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
        ])
        for worker in workers:
            worker.complete(batch_fixtures._reference(123))
        for key in self.objects._current_inpaint_keys()[:2]:
            self.assertFalse(self.objects.get_inpaint_tried_reference_views(key))

    def test_preset_prompt_changes_keep_per_object_selection_and_use_latest_text(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[3]
        _select(self.workspace, preset.preset_id)
        self.workspace.next_inpaint_button.click()
        edited = replace(preset, name="Edited side view", prompt="New reusable side-view prompt")
        views = tuple(edited if item.preset_id == edited.preset_id else item for item in DEFAULT_REFERENCE_VIEW_PRESETS)
        self.workspace.set_reference_view_presets(views)
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), edited.preset_id)
        self.assertEqual(self.workspace._selected_reference_generation_prompt(), edited.prompt)

    def test_preset_settings_refresh_untouched_blobs_but_keep_edited_offscreen_overrides(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        keys = self.objects._current_inpaint_keys()
        override = _prompt(preset.prompt, "Keep the original upholstery")
        self.workspace.reference_generation_prompt_edit.setText(override)
        self.workspace.next_inpaint_button.click()
        self.workspace.next_inpaint_button.click()
        edited = replace(preset, prompt="A new reusable camera-view prompt")
        self.workspace.set_reference_view_presets((edited, *DEFAULT_REFERENCE_VIEW_PRESETS[1:]))
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), edited.prompt)
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), edited.prompt)
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), preset.preset_id)
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), override)
        self.assertEqual(self.objects.get_inpaint_reference_view_preferences(keys[0]), (preset.preset_id, override))

    def test_selecting_another_view_replaces_only_the_selected_blobs_edited_prompt(self):
        presets = DEFAULT_REFERENCE_VIEW_PRESETS[:2]
        override = "Recreate the furniture with no loose contents"
        self.workspace.reference_generation_prompt_edit.setText(override)
        self.workspace.next_inpaint_button.click()
        _select(self.workspace, presets[1].preset_id)
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), presets[1].prompt)
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), override)
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), presets[0].preset_id)
        _select(self.workspace, presets[1].preset_id)
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), presets[1].prompt)


# ### Grouped local request integration ###
class LocalInpaintReferenceViewWorkspaceTests(unittest.TestCase):
    setUp = local_fixtures.LocalReferenceBatchWorkspaceTests.setUp
    tearDown = local_fixtures.LocalReferenceBatchWorkspaceTests.tearDown
    _paint = local_fixtures.LocalReferenceBatchWorkspaceTests._paint
    _start_local = local_fixtures.LocalReferenceBatchWorkspaceTests._start_local

    def test_local_batch_items_freeze_individual_prompts_and_mark_matching_tried_views(self):
        presets = DEFAULT_REFERENCE_VIEW_PRESETS[1:4]
        for preset in presets:
            _select(self.workspace, preset.preset_id)
            self.workspace.next_inpaint_button.click()
        extras = ("Keep its distinctive details", "Repair the missing leg", "")
        for preset, extra in zip(presets, extras):
            self.workspace.reference_generation_prompt_edit.setText(_prompt(preset.prompt, extra))
            self.workspace.next_inpaint_button.click()
        worker = self._start_local()
        expected = [_prompt(preset.prompt, extra) for preset, extra in zip(presets, extras)]
        self.assertEqual([item.prompt for item in worker.items], expected)
        self.workspace.reference_generation_prompt_edit.setText("A later instruction")
        _select(self.workspace, DEFAULT_REFERENCE_VIEW_PRESETS[0].preset_id)
        for index, value in enumerate((101, 102, 103)):
            worker.render(index, value)
            worker.decide(index, True)
        worker.complete()
        self.assertEqual([item.prompt for item in worker.items], expected)
        for key, preset in zip(self.objects._current_inpaint_keys(), presets):
            self.assertEqual(
                self.objects.get_inpaint_tried_reference_views(key),
                frozenset({(preset.preset_id, preset.prompt)}),
            )
        self.assertEqual(len(local_fixtures._ControlledLocalReferenceBatchThread.created), 1)

    def test_local_batch_distinguishes_custom_prompt_from_other_objects_default_view(self):
        _select(self.workspace, "")
        self.workspace.reference_generation_prompt_edit.setText("A customized view for only the first object")
        worker = self._start_local()
        self.assertEqual([item.prompt for item in worker.items], [
            "A customized view for only the first object",
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
        ])
        for index in range(3):
            worker.render(index, 101 + index)
            worker.decide(index, False)
        worker.complete()
        self.assertEqual(_label(self.workspace, ""), "(Write prompt)")
        self.assertFalse(self.objects.get_inpaint_tried_reference_views(self.objects._current_inpaint_keys()[0]))


# ### Resident-worker per-item prompt dispatch ###
class LocalPerItemReferencePromptWorkerTests(unittest.TestCase):
    setUp = worker_fixtures.LocalReferenceBatchWorkerTests.setUp
    tearDown = worker_fixtures.LocalReferenceBatchWorkerTests.tearDown

    def test_resident_image_model_receives_each_items_prompt_with_shared_fallback(self):
        self.thread.items = tuple(
            replace(item, prompt=prompt)
            for item, prompt in zip(self.thread.items, ("First object's front view", None, "Third object's side view"))
        )
        received = []
        generate = self.sessions.generate

        def capture(image, prompt, **kwargs):
            received.append((int(image[0, 0, 0]), prompt))
            return generate(image, prompt, **kwargs)

        with patch.object(self.sessions, "generate", side_effect=capture):
            self.thread.run()
        self.assertEqual(received, [
            (10, "First object's front view"),
            (20, "Show the object from the front"),
            (30, "Third object's side view"),
        ])
        self.assertEqual(self.sessions.events.count("image open"), 1)
        self.assertEqual(self.sessions.events.count("symmetry open"), 1)
        self.assertEqual(len(self.finished), 3)


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
