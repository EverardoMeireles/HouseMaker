"""Unified reference prompts without model loading or paid API requests."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import test_all_reference_generation as batch_fixtures
import test_reference_generation_workspace as reference_fixtures
from PySide6.QtWidgets import QLabel

from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.merged_generation_workspace import (
    REFERENCE_GENERATION_PROMPT_SETTINGS_KEY,
    REFERENCE_GENERATION_VIEW_SETTINGS_KEY,
)
from housemaker.reference_view_presets import DEFAULT_REFERENCE_VIEW_PRESETS

# ### Migration keys ###
LEGACY_CUSTOM_PROMPT_SETTINGS_KEY = "generation/reference_generation_prompt"
LEGACY_AI_PROMPT_SETTINGS_KEY = "generation/reference_generation_ai_prompt"


# ### Single-reference prompt tests ###
class ReferenceGenerationPromptTests(unittest.TestCase):
    setUp = reference_fixtures.ReferenceGenerationWorkspaceTests.setUp
    tearDown = reference_fixtures.ReferenceGenerationWorkspaceTests.tearDown
    _apply_reference_generation_model = (
        reference_fixtures.ReferenceGenerationWorkspaceTests._apply_reference_generation_model
    )
    _reopened_workspace = (
        reference_fixtures.ReferenceGenerationWorkspaceTests._reopened_workspace
    )

    def _start_reference(self):
        self.workspace._activate_temporary_object_reference(
            reference_fixtures._opaque_reference(), source_label="Loaded image"
        )
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        self.workspace.generate_reference_button.click()
        self.assertIsNotNone(self.workspace._reference_generation_runtime)
        return reference_fixtures._ControlledReferenceThread.created[-1]

    def _seed_legacy_preferences(self, view_id: str, custom: str, extra: str):
        self.settings.remove(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY)
        self.settings.set(REFERENCE_GENERATION_VIEW_SETTINGS_KEY, view_id)
        self.settings.set(LEGACY_CUSTOM_PROMPT_SETTINGS_KEY, custom)
        self.settings.set(LEGACY_AI_PROMPT_SETTINGS_KEY, extra)

    def _assert_long_prompt_can_load_first_blob(self, workspace, expected: str):
        self.assertEqual(workspace.reference_generation_prompt_edit.text(), expected)
        self.assertEqual(
            workspace._default_inpaint_reference_view_preferences[1], expected
        )
        warning = workspace.reference_generation_status_label.text()
        self.assertIn("4,000", warning)
        self.assertIn("unchanged", warning)

        video = self.root / "long-prompt.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
        )
        self.assertTrue(writer.isOpened())
        try:
            writer.write(np.full((48, 64, 3), 80, dtype=np.uint8))
        finally:
            writer.release()
        workspace.object_workspace.load_video(str(video))
        workspace.surface_workspace.load_video(str(video))
        workspace.video_view.set_strokes(
            [MaskStroke(MASK_MODE_PAINT, 0.2, (MaskPoint(0.5, 0.5),))]
        )
        workspace.video_view.strokes_changed.emit(workspace.video_view.get_strokes())
        workspace.sync_shared_controls()
        self.assertEqual(workspace.reference_generation_prompt_edit.text(), expected)
        self.assertIsNotNone(workspace.object_workspace.current_inpaint_reference_key())

    def test_first_preset_populates_prompt_and_redundant_field_is_gone(self):
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(),
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
        )
        self.assertFalse(hasattr(self.workspace, "reference_generation_ai_prompt_edit"))

    def test_reference_prompt_is_visible_for_all_views(self):
        combo = self.workspace.reference_generation_view_combo
        for index in range(combo.count()):
            with self.subTest(view=combo.itemData(index)):
                combo.setCurrentIndex(index)
                self.assertTrue(
                    self.workspace.reference_generation_prompt_edit.isVisibleTo(
                        self.workspace.reference_generation_section
                    )
                )

    def test_editing_selected_view_changes_request_without_mutating_preset(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[1]
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(preset.preset_id))
        self.workspace.reference_generation_prompt_edit.setText(
            "  Show a clean frontal view without the fabric cover  "
        )

        worker = self._start_reference()

        self.assertEqual(
            worker.prompt, "Show a clean frontal view without the fabric cover"
        )
        self.assertEqual(combo.currentData(), preset.preset_id)
        self.assertEqual(DEFAULT_REFERENCE_VIEW_PRESETS[1], preset)
        self.assertEqual(self.workspace._reference_view_presets[1], preset)

    def test_selecting_another_view_populates_its_prompt(self):
        self.workspace.reference_generation_prompt_edit.setText("Edited view")
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[2]
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(preset.preset_id))
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(), preset.prompt
        )
        combo.setCurrentIndex(combo.findData(""))
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(), preset.prompt
        )

    def test_blank_prompt_is_not_silently_replaced_by_preset(self):
        self.workspace.reference_generation_prompt_edit.clear()
        self.workspace._activate_temporary_object_reference(
            reference_fixtures._opaque_reference(), source_label="Loaded image"
        )
        self.assertEqual(self.workspace._selected_reference_generation_prompt(), "")
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())

    def test_reference_and_texture_prompts_are_independent(self):
        self.workspace.ai_prompt_edit.setText("Meshy material instructions")
        self.workspace.reference_generation_prompt_edit.setText("Front view")

        worker = self._start_reference()

        self.assertEqual(worker.prompt, "Front view")
        self.assertEqual(self.workspace.ai_prompt_edit.text(), "Meshy material instructions")
        self.assertEqual(
            self.workspace.ai_prompt_edit.placeholderText(), "Texture prompt (optional)"
        )
        labels = self.workspace.shared_material_section.findChildren(QLabel)
        self.assertIn("Texture prompt", [label.text() for label in labels])
        self.assertNotIn("AI prompt", [label.text() for label in labels])

    def test_edited_preset_prompt_is_persisted_and_restored(self):
        self.workspace.reference_generation_prompt_edit.setText("My edited reference")
        self.assertEqual(
            self.settings.get(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY),
            "My edited reference",
        )
        with self._reopened_workspace() as reopened:
            self.assertEqual(
                reopened.reference_generation_prompt_edit.text(), "My edited reference"
            )
            self.assertEqual(
                reopened._selected_reference_generation_prompt(), "My edited reference"
            )

    def test_legacy_preset_and_extra_are_combined_once(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[1]
        self._seed_legacy_preferences(
            preset.preset_id, "Unused hidden custom prompt", "  Remove the cover  "
        )
        expected = f"{preset.prompt}, Remove the cover"

        for reopening in range(2):
            with self.subTest(reopening=reopening), self._reopened_workspace() as reopened:
                self.assertEqual(
                    reopened.reference_generation_prompt_edit.text(), expected
                )
                self.assertEqual(
                    reopened._selected_reference_generation_prompt(), expected
                )
    def test_legacy_custom_and_extra_are_combined_once(self):
        self._seed_legacy_preferences("", "  A clean front view  ", "Remove clutter")
        for _ in range(2):
            with self._reopened_workspace() as reopened:
                self.assertEqual(
                    reopened.reference_generation_prompt_edit.text(),
                    "A clean front view, Remove clutter",
                )

    def test_canonical_prompt_takes_priority_over_both_legacy_fields(self):
        self._seed_legacy_preferences("front", "Old custom", "Old extra")
        self.settings.set(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY, "Already migrated")
        with self._reopened_workspace() as reopened:
            self.assertEqual(
                reopened.reference_generation_prompt_edit.text(), "Already migrated"
            )

    def test_canonical_explicit_blank_is_preserved_on_reopening(self):
        self._seed_legacy_preferences("front", "Old custom", "Old extra")
        self.settings.set(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY, "")
        with self._reopened_workspace() as reopened:
            self.assertEqual(reopened.reference_generation_prompt_edit.text(), "")
            self.assertEqual(reopened._selected_reference_generation_prompt(), "")

    def test_oversized_legacy_prompts_are_clamped_without_overwriting_saved_text(self):
        custom = "A" * 3_500
        extra = "B" * 1_000
        self._seed_legacy_preferences("", custom, extra)
        expected = f"{custom}, {extra}"[:4_000]

        with self._reopened_workspace() as reopened:
            self._assert_long_prompt_can_load_first_blob(reopened, expected)

        self.assertEqual(self.settings.get(LEGACY_CUSTOM_PROMPT_SETTINGS_KEY), custom)
        self.assertEqual(self.settings.get(LEGACY_AI_PROMPT_SETTINGS_KEY), extra)
        self.assertIsNone(self.settings.get(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY))

    def test_oversized_canonical_prompt_is_clamped_without_overwriting_saved_text(self):
        original = "Detailed reference instruction. " * 200
        self.settings.set(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY, original)

        with self._reopened_workspace() as reopened:
            self._assert_long_prompt_can_load_first_blob(reopened, original[:4_000])

        self.assertEqual(
            self.settings.get(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY), original
        )

    def test_prompt_is_locked_while_running_and_request_is_a_snapshot(self):
        edit = self.workspace.reference_generation_prompt_edit
        edit.setText("A clean view without the cover")
        worker = self._start_reference()
        self.assertFalse(edit.isEnabled())

        edit.setText("New instructions for the next request")
        self.assertEqual(worker.prompt, "A clean view without the cover")
        worker.complete(reference_fixtures._opaque_reference(190))

        self.assertTrue(edit.isEnabled())

    def test_cancelling_restores_prompt_field_without_erasing_text(self):
        edit = self.workspace.reference_generation_prompt_edit
        edit.setText("Keep this edited view")
        self._start_reference()
        self.workspace.generate_reference_button.click()

        self.assertTrue(edit.isEnabled())
        self.assertEqual(edit.text(), "Keep this edited view")


# ### Cross-frame batch prompt snapshots ###
class AllReferenceGenerationPromptTests(unittest.TestCase):
    setUp = batch_fixtures.AllReferenceGenerationTests.setUp
    tearDown = batch_fixtures.AllReferenceGenerationTests.tearDown
    _paint = batch_fixtures.AllReferenceGenerationTests._paint
    _start_all = batch_fixtures.AllReferenceGenerationTests._start_all
    _finish_all = batch_fixtures.AllReferenceGenerationTests._finish_all

    def test_each_blob_captures_its_edited_prompt_before_queue_starts(self):
        edit = self.workspace.reference_generation_prompt_edit
        edit.setText("Front view without any fabric covers")

        workers = self._start_all()

        self.assertFalse(edit.isEnabled())
        expected_prompts = [
            "Front view without any fabric covers",
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
        ]
        self.assertEqual([worker.prompt for worker in workers], expected_prompts)
        edit.setText("Different instructions for future requests")
        self.assertEqual([worker.prompt for worker in workers], expected_prompts)

        self._finish_all(workers)

        self.assertTrue(edit.isEnabled())
        workers = self._start_all()
        self.assertEqual(
            [worker.prompt for worker in workers],
            [
                "Different instructions for future requests",
                DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
                DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
            ],
        )


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
