"""Optional furniture clearing belongs to each reference request and blob."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import test_all_reference_generation as batch_fixtures
import test_local_reference_batch_workspace as local_fixtures
import test_reference_generation_workspace as single_fixtures
from PySide6.QtWidgets import QCheckBox

from housemaker.reference_generation import REMOVE_OBJECTS_FROM_FURNITURE_PROMPT
from housemaker.reference_view_presets import DEFAULT_REFERENCE_VIEW_PRESETS

# ### Shared prompt and control helpers ###
SETTINGS_KEY = "generation/remove_objects_from_furniture"


def _prompt(base: str, extra: str = "", *, remove: bool = False) -> str:
    return ", ".join(
        part
        for part in (
            base,
            extra,
            REMOVE_OBJECTS_FROM_FURNITURE_PROMPT if remove else "",
        )
        if part
    )


def _checkbox(workspace) -> QCheckBox:
    return workspace.remove_objects_from_furniture_checkbox


# ### Standalone controls, persistence, and single requests ###
class ReferenceFurnitureRemovalWorkspaceTests(unittest.TestCase):
    setUp = single_fixtures.ReferenceGenerationWorkspaceTests.setUp
    tearDown = single_fixtures.ReferenceGenerationWorkspaceTests.tearDown
    _apply_reference_generation_model = (
        single_fixtures.ReferenceGenerationWorkspaceTests._apply_reference_generation_model
    )
    _reopened_workspace = (
        single_fixtures.ReferenceGenerationWorkspaceTests._reopened_workspace
    )

    def test_checkbox_defaults_off_directly_below_view_above_custom_prompt(self):
        checkbox = _checkbox(self.workspace)
        self.assertIsInstance(checkbox, QCheckBox)
        self.assertEqual(checkbox.text(), "Remove objects from furniture")
        self.assertFalse(checkbox.isChecked())
        layout = self.workspace.reference_generation_section.layout()
        view_control = self.workspace.reference_generation_view_combo.parentWidget()
        self.assertEqual(layout.indexOf(checkbox), layout.indexOf(view_control) + 1)
        self.assertEqual(
            layout.indexOf(self.workspace.reference_generation_prompt_control),
            layout.indexOf(checkbox) + 1,
        )

    def test_single_saved_view_appends_removal_after_full_prompt_without_mutating_inputs(self):
        source = single_fixtures._opaque_reference()
        self.workspace._activate_temporary_object_reference(source, source_label="Test")
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(preset.preset_id))
        prompt = _prompt(preset.prompt, "Keep the wood color")
        self.workspace.reference_generation_prompt_edit.setText(prompt)
        _checkbox(self.workspace).setChecked(True)
        self.workspace.generate_reference_button.click()
        worker = single_fixtures._ControlledReferenceThread.created[-1]
        self.assertEqual(worker.prompt, _prompt(preset.prompt, "Keep the wood color", remove=True))
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), prompt)
        self.assertEqual(self.workspace._reference_view_presets[0].prompt, preset.prompt)
        self.assertFalse(_checkbox(self.workspace).isEnabled())
        self.assertFalse(combo.isEnabled())
        worker.complete(single_fixtures._opaque_reference(90))
        self.assertTrue(_checkbox(self.workspace).isEnabled())
        self.assertTrue(combo.isEnabled())

    def test_custom_prompt_adds_removal_once_and_unchecking_restores_original(self):
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(""))
        prompt = "Show this object from the front, Preserve its material"
        self.workspace.reference_generation_prompt_edit.setText(prompt)
        checkbox = _checkbox(self.workspace)
        checkbox.setChecked(True)
        checkbox.setChecked(False)
        checkbox.setChecked(True)
        expected = _prompt("Show this object from the front", "Preserve its material", remove=True)
        self.assertEqual(self.workspace._selected_reference_generation_prompt(), expected)
        self.assertEqual(expected.count(REMOVE_OBJECTS_FROM_FURNITURE_PROMPT), 1)
        self.assertEqual(self.workspace.reference_generation_prompt_edit.text(), prompt)
        checkbox.setChecked(False)
        self.assertEqual(
            self.workspace._selected_reference_generation_prompt(),
            "Show this object from the front, Preserve its material",
        )

    def test_single_request_without_removal_keeps_existing_prompt(self):
        self.workspace._activate_temporary_object_reference(
            single_fixtures._opaque_reference(), source_label="Test",
        )
        self.workspace.reference_generation_prompt_edit.setText(
            _prompt(DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt, "Preserve the carved details")
        )
        self.workspace.generate_reference_button.click()
        worker = single_fixtures._ControlledReferenceThread.created[-1]
        self.assertEqual(worker.prompt, _prompt(DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt, "Preserve the carved details"))
        self.assertNotIn(REMOVE_OBJECTS_FROM_FURNITURE_PROMPT, worker.prompt)
        worker.complete(error="Controlled provider failure")
        self.assertTrue(_checkbox(self.workspace).isEnabled())

    def test_standalone_checkbox_is_persisted_and_restored_when_reopened(self):
        self.assertIsNone(self.objects.current_inpaint_reference_key())
        _checkbox(self.workspace).setChecked(True)
        self.assertIs(self.settings.get(SETTINGS_KEY), True)
        with self._reopened_workspace() as reopened:
            self.assertTrue(_checkbox(reopened).isChecked())
            _checkbox(reopened).setChecked(False)
        with self._reopened_workspace() as reopened:
            self.assertFalse(_checkbox(reopened).isChecked())

    def test_malformed_persisted_checkbox_values_default_off(self):
        for value in (None, "true", "false", 1, [], {}):
            with self.subTest(value=value):
                self.settings.set(SETTINGS_KEY, value)
                with self._reopened_workspace() as reopened:
                    self.assertFalse(_checkbox(reopened).isChecked())


# ### Per-blob navigation and frozen API batches ###
class InpaintFurnitureRemovalWorkspaceTests(unittest.TestCase):
    setUp = batch_fixtures.AllReferenceGenerationTests.setUp
    tearDown = batch_fixtures.AllReferenceGenerationTests.tearDown
    _paint = batch_fixtures.AllReferenceGenerationTests._paint
    _start_all = batch_fixtures.AllReferenceGenerationTests._start_all
    _finish_all = batch_fixtures.AllReferenceGenerationTests._finish_all

    def _seed_flags(self):
        flags = (True, False, True)
        for enabled in flags:
            _checkbox(self.workspace).setChecked(enabled)
            self.workspace.next_inpaint_button.click()
        return flags

    def test_navigation_restores_flags_independently_across_frames(self):
        flags = self._seed_flags()
        for enabled in flags:
            self.assertEqual(_checkbox(self.workspace).isChecked(), enabled)
            self.workspace.next_inpaint_button.click()
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.objects._displayed_frame_index, 2)
        self.assertTrue(_checkbox(self.workspace).isChecked())
        keys = self.objects._current_inpaint_keys()
        self.assertEqual(
            [self.objects.get_inpaint_remove_objects_from_furniture(key) for key in keys],
            list(flags),
        )

    def test_direct_selection_and_seek_restore_the_selected_blobs_flag(self):
        self._seed_flags()
        for frame, blob_index, expected in ((0, 1, False), (2, 0, True), (0, 0, True)):
            with self.subTest(frame=frame, blob_index=blob_index):
                self.workspace.seekbar.setValue(frame)
                self.workspace.video_view.select_blob(blob_index)
                self.assertEqual(_checkbox(self.workspace).isChecked(), expected)

    def test_existing_and_new_blobs_do_not_inherit_previous_blobs_flag(self):
        _checkbox(self.workspace).setChecked(True)
        self.workspace.next_inpaint_button.click()
        self.assertFalse(_checkbox(self.workspace).isChecked())
        self.workspace.next_inpaint_button.click()
        self.assertFalse(_checkbox(self.workspace).isChecked())
        self._paint(1, 0.5)
        self.assertIsNotNone(self.objects.current_inpaint_reference_key())
        self.assertFalse(_checkbox(self.workspace).isChecked())

    def test_empty_frame_does_not_promote_previous_blobs_flag_to_new_defaults(self):
        _checkbox(self.workspace).setChecked(True)
        self.workspace.seekbar.setValue(1)
        self.assertIsNone(self.objects.current_inpaint_reference_key())
        self.assertFalse(_checkbox(self.workspace).isChecked())
        self.workspace.reference_generation_strength_spinbox.setValue(0.60)
        self.workspace.seekbar.setValue(0)
        self.workspace.video_view.select_blob(1)
        self.assertFalse(_checkbox(self.workspace).isChecked())

    def test_programmatic_blob_flag_is_read_back_without_changing_other_blobs(self):
        first, second, _third = self.objects._current_inpaint_keys()
        self.assertTrue(self.objects.set_inpaint_remove_objects_from_furniture(first, True))
        self.assertTrue(self.objects.get_inpaint_remove_objects_from_furniture(first))
        self.assertIsNot(self.objects.get_inpaint_remove_objects_from_furniture(second), True)
        self.workspace.next_inpaint_button.click()
        self.assertFalse(_checkbox(self.workspace).isChecked())
        self.workspace.previous_inpaint_button.click()
        self.assertTrue(_checkbox(self.workspace).isChecked())

    def test_unknown_or_removed_blob_cannot_receive_a_preference(self):
        invalid_key = (999, "not-a-current-blob")
        self.assertIsNone(self.objects.get_inpaint_remove_objects_from_furniture(invalid_key))
        self.assertFalse(self.objects.set_inpaint_remove_objects_from_furniture(invalid_key, True))
        key = self.objects.current_inpaint_reference_key()
        self.assertTrue(self.objects.set_inpaint_remove_objects_from_furniture(key, True))
        self.objects.set_data(None)
        self.assertIsNone(self.objects.get_inpaint_remove_objects_from_furniture(key))
        self.assertFalse(self.objects.set_inpaint_remove_objects_from_furniture(key, True))

    def test_single_generation_uses_only_selected_blobs_flag(self):
        self._seed_flags()
        self.workspace.next_inpaint_button.click()
        self.workspace.generate_reference_button.click()
        worker = batch_fixtures._ControlledReferenceThread.created[-1]
        self.assertEqual(worker.prompt, DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt)
        worker.complete(batch_fixtures._reference(101))
        self.workspace.previous_inpaint_button.click()
        self.workspace.generate_reference_button.click()
        worker = batch_fixtures._ControlledReferenceThread.created[-1]
        self.assertEqual(worker.prompt, _prompt(DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt, remove=True))
        worker.complete(batch_fixtures._reference(102))

    def test_api_batch_freezes_flags_and_appends_after_individual_full_prompts(self):
        flags = self._seed_flags()
        extras = ("Preserve its color", "Keep the upholstery", "")
        for extra in extras:
            self.workspace.reference_generation_prompt_edit.setText(
                _prompt(DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt, extra)
            )
            self.workspace.next_inpaint_button.click()
        key = self.objects.current_inpaint_reference_key()
        workers = self._start_all()
        expected = [
            _prompt(DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt, extra, remove=enabled)
            for extra, enabled in zip(extras, flags)
        ]
        self.assertEqual([worker.prompt for worker in workers], expected)
        self.assertEqual(self.objects.current_inpaint_reference_key(), key)
        self.assertFalse(_checkbox(self.workspace).isEnabled())
        _checkbox(self.workspace).setChecked(False)
        self.assertEqual([worker.prompt for worker in workers], expected)
        self._finish_all(workers)
        self.assertTrue(_checkbox(self.workspace).isEnabled())


# ### Resident local batch ownership ###
class LocalFurnitureRemovalWorkspaceTests(unittest.TestCase):
    setUp = local_fixtures.LocalReferenceBatchWorkspaceTests.setUp
    tearDown = local_fixtures.LocalReferenceBatchWorkspaceTests.tearDown
    _paint = local_fixtures.LocalReferenceBatchWorkspaceTests._paint
    _start_local = local_fixtures.LocalReferenceBatchWorkspaceTests._start_local
    _finish_local = local_fixtures.LocalReferenceBatchWorkspaceTests._finish_local

    def test_local_batch_freezes_each_blobs_prompt_with_optional_removal(self):
        prompts = (
            "A clean front view with the existing wood finish",
            "A low three-quarter view with the original upholstery",
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
        )
        for prompt, enabled in zip(prompts, (True, False, True)):
            self.workspace.reference_generation_prompt_edit.setText(prompt)
            _checkbox(self.workspace).setChecked(enabled)
            self.workspace.next_inpaint_button.click()
        worker = self._start_local()
        expected = [
            _prompt(prompt, remove=enabled)
            for prompt, enabled in zip(prompts, (True, False, True))
        ]
        self.assertEqual([item.prompt for item in worker.items], expected)
        self.assertFalse(_checkbox(self.workspace).isEnabled())
        self.workspace.reference_generation_prompt_edit.setText("A later full-prompt edit")
        _checkbox(self.workspace).setChecked(False)
        self.assertEqual([item.prompt for item in worker.items], expected)
        self._finish_local(worker)
        self.assertTrue(_checkbox(self.workspace).isEnabled())


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
