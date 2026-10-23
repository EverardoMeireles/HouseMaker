"""Prompt drafts must not do filesystem, Atlas or mesh work per keystroke."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest

from housemaker.generation_state import MaskPoint, MaskStroke
from housemaker.merged_generation_workspace import (
    REFERENCE_GENERATION_PROMPT_SETTINGS_KEY,
)
from tests import test_all_reference_generation as batch_fixtures
from tests import test_reference_generation_workspace as reference_fixtures
from tests.test_reference_generation_workspace import (
    _ControlledReferenceThread,
    _opaque_reference,
)


# ### Shared local-only fixture ###
class ReferencePromptTypingTests(unittest.TestCase):
    setUp = reference_fixtures.ReferenceGenerationWorkspaceTests.setUp
    tearDown = reference_fixtures.ReferenceGenerationWorkspaceTests.tearDown
    _apply_reference_generation_model = reference_fixtures.ReferenceGenerationWorkspaceTests._apply_reference_generation_model
    _load_video = reference_fixtures.ReferenceGenerationWorkspaceTests._load_video

    def _select_custom_prompt(self) -> None:
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(""))
        self.workspace.reference_generation_prompt_edit.clear()

    def _activate_blob_reference(self) -> None:
        self._load_video()
        self.workspace._activate_temporary_object_reference(
            _opaque_reference(),
            source_label="Loaded image",
        )
        self._select_custom_prompt()

    # ### Typing costs and explicit commit boundaries ###
    def test_blob_prompt_keystrokes_do_not_write_clone_notify_or_fully_sync(
        self,
    ) -> None:
        self._activate_blob_reference()
        atlas_listener = Mock()
        self.objects.data_changed.connect(atlas_listener)
        with (
            patch.object(self.settings, "set") as settings_write,
            patch.object(self.settings, "_write") as file_write,
            patch.object(
                self.objects,
                "get_data",
                side_effect=AssertionError("No project snapshot"),
            ),
            patch.object(
                self.workspace, "_sync_reference_generation_controls"
            ) as full_sync,
        ):
            QTest.keyClicks(
                self.workspace.reference_generation_prompt_edit,
                "Remove every loose object",
            )
        settings_write.assert_not_called()
        file_write.assert_not_called()
        full_sync.assert_not_called()
        atlas_listener.assert_not_called()
        self.assertEqual(
            self.objects.get_inpaint_reference_view_preferences(
                self.objects.current_inpaint_reference_key()
            ),
            ("", "Remove every loose object"),
        )

    def test_ordinary_prompt_keystrokes_only_update_the_current_blob(self) -> None:
        self._load_video()
        self._select_custom_prompt()
        key = self.objects.current_inpaint_reference_key()
        with (
            patch.object(self.settings, "set") as settings_write,
            patch.object(
                self.workspace, "_sync_reference_generation_controls"
            ) as full_sync,
        ):
            QTest.keyClicks(
                self.workspace.reference_generation_prompt_edit, "A clearer front view"
            )
        settings_write.assert_not_called()
        full_sync.assert_not_called()
        self.assertEqual(
            self.objects.get_inpaint_reference_view_preferences(key),
            ("", "A clearer front view"),
        )

    def test_negative_prompt_typing_also_avoids_settings_and_scene_notifications(
        self,
    ) -> None:
        self._activate_blob_reference()
        self.workspace.reference_generation_negative_prompt_edit.clear()
        changed = Mock()
        self.objects.data_changed.connect(changed)
        with patch.object(self.settings, "set") as settings_write:
            QTest.keyClicks(
                self.workspace.reference_generation_negative_prompt_edit,
                "blurry, clutter",
            )
        settings_write.assert_not_called()
        changed.assert_not_called()

    def test_editing_finished_commits_defaults_without_scene_notifications(
        self,
    ) -> None:
        self._select_custom_prompt()
        changed = Mock()
        self.objects.data_changed.connect(changed)
        QTest.keyClicks(
            self.workspace.reference_generation_prompt_edit, "A custom chair"
        )
        self.assertNotEqual(
            self.settings.get(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY),
            "A custom chair",
        )
        self.workspace.reference_generation_prompt_edit.editingFinished.emit()
        self.assertEqual(
            self.settings.get(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY),
            "A custom chair",
        )
        changed.assert_not_called()
        with patch.object(self.settings, "set") as settings_write:
            self.workspace.reference_generation_prompt_edit.editingFinished.emit()
        settings_write.assert_not_called()

    def test_same_blob_preferences_do_not_notify_the_scene(self) -> None:
        self._activate_blob_reference()
        self.workspace.reference_generation_prompt_edit.setText("A draft")
        key = self.objects.current_inpaint_reference_key()
        current_preferences = self.objects.get_inpaint_reference_view_preferences(key)
        changed = Mock()
        self.objects.data_changed.connect(changed)
        self.objects.set_inpaint_reference_view_preferences(key, "", "A draft")
        self.assertEqual(
            self.objects.get_inpaint_reference_view_preferences(key),
            current_preferences,
        )
        changed.assert_not_called()

    def test_late_text_and_focus_callbacks_after_shutdown_are_noops(self) -> None:
        self._activate_blob_reference()
        self.workspace.reference_generation_prompt_edit.setText("Last valid draft")
        self.workspace.shutdown()
        key = self.objects.current_inpaint_reference_key()
        preferences = self.objects.get_inpaint_reference_view_preferences(key)
        changed = Mock()
        self.objects.data_changed.connect(changed)
        with (
            patch.object(self.settings, "set") as settings_write,
            patch.object(
                self.workspace, "_sync_reference_generation_controls"
            ) as full_sync,
        ):
            self.workspace.reference_generation_prompt_edit.setText("Late Qt update")
            self.workspace.reference_generation_prompt_edit.editingFinished.emit()
            self.workspace.flush_reference_generation_preferences()
        self.assertEqual(
            self.objects.get_inpaint_reference_view_preferences(key), preferences
        )
        settings_write.assert_not_called()
        full_sync.assert_not_called()
        changed.assert_not_called()

    def test_shutdown_still_cancels_workers_if_best_effort_settings_flush_fails(
        self,
    ) -> None:
        self._activate_blob_reference()
        self.workspace.reference_generation_prompt_edit.setText("A reference request")
        self.workspace.generate_reference_button.click()
        worker = _ControlledReferenceThread.created[-1]
        self.workspace.reference_generation_prompt_edit.setText("Later draft")
        with patch.object(
            self.workspace,
            "flush_reference_generation_preferences",
            side_effect=OSError("Unavailable settings"),
        ):
            self.workspace.shutdown()
        self.assertTrue(self.workspace._is_shutdown)
        self.assertTrue(worker.interrupted)
        self.assertIsNone(self.workspace._reference_generation_runtime)

    # ### Immediate controls, job snapshots and draft restoration ###
    def test_empty_and_nonempty_prompts_update_action_buttons_immediately(self) -> None:
        self._load_video()
        self._select_custom_prompt()
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        self.assertFalse(self.workspace.generate_all_references_button.isEnabled())
        QTest.keyClicks(self.workspace.reference_generation_prompt_edit, "Chair")
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        self.assertTrue(self.workspace.generate_all_references_button.isEnabled())
        self.workspace.reference_generation_prompt_edit.setText("   ")
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        self.assertFalse(self.workspace.generate_all_references_button.isEnabled())

    def test_typing_without_an_image_does_not_enable_generation(self) -> None:
        self._select_custom_prompt()
        QTest.keyClicks(self.workspace.reference_generation_prompt_edit, "Chair")
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())

    def test_typing_during_a_pending_batch_cannot_enable_another_generation(
        self,
    ) -> None:
        self._load_video()
        with patch.object(
            self.workspace,
            "_batch_reference_generation_runtimes",
            {"pending": object()},
        ):
            self.workspace._sync_reference_generation_controls()
            self.workspace.reference_generation_prompt_edit.setText("A changed draft")
            self.assertFalse(self.workspace.generate_reference_button.isEnabled())
            self.assertTrue(self.workspace.generate_all_references_button.isEnabled())

    def test_generate_reference_freezes_the_latest_visible_text_without_focus_loss(
        self,
    ) -> None:
        self._activate_blob_reference()
        changed = Mock()
        self.objects.data_changed.connect(changed)
        QTest.keyClicks(
            self.workspace.reference_generation_prompt_edit, "Latest complete prompt"
        )
        self.workspace.reference_generation_prompt_edit.editingFinished.emit()
        self.workspace.generate_reference_button.click()
        self.assertEqual(
            _ControlledReferenceThread.created[-1].prompt, "Latest complete prompt"
        )
        self.assertEqual(len(_ControlledReferenceThread.created), 1)
        changed.assert_not_called()

    def test_project_snapshot_keeps_the_latest_live_blob_draft(self) -> None:
        self._activate_blob_reference()
        QTest.keyClicks(
            self.workspace.reference_generation_prompt_edit,
            "Unsaved draft in the project",
        )
        key = self.objects.current_inpaint_reference_key()
        self.objects.get_data().to_dict()
        self.assertEqual(
            self.objects.get_inpaint_reference_view_preferences(key),
            ("", "Unsaved draft in the project"),
        )
        self.workspace.sync_shared_controls()
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(),
            "Unsaved draft in the project",
        )

    def test_ordinary_blob_navigation_keeps_uncommitted_drafts_independent(
        self,
    ) -> None:
        self._load_video()
        strokes = [
            MaskStroke("paint", 0.07, (MaskPoint(0.25, 0.5),)),
            MaskStroke("paint", 0.07, (MaskPoint(0.75, 0.5),)),
        ]
        self.workspace.video_view.set_strokes(strokes)
        self.workspace.video_view.strokes_changed.emit(strokes)
        self.workspace.sync_shared_controls()
        self._select_custom_prompt()
        QTest.keyClicks(self.workspace.reference_generation_prompt_edit, "First draft")
        self.workspace.next_inpaint_button.click()
        self._select_custom_prompt()
        QTest.keyClicks(self.workspace.reference_generation_prompt_edit, "Second draft")
        self.workspace.previous_inpaint_button.click()
        self.objects.get_data()
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(), "First draft"
        )
        self.workspace.next_inpaint_button.click()
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(), "Second draft"
        )


# ### Batch snapshots without a provider or worker run ###
class BatchReferencePromptTypingTests(unittest.TestCase):
    setUp = batch_fixtures.AllReferenceGenerationTests.setUp
    tearDown = batch_fixtures.AllReferenceGenerationTests.tearDown
    _paint = batch_fixtures.AllReferenceGenerationTests._paint

    def test_generate_all_commits_the_latest_visible_text_before_snapshotting(
        self,
    ) -> None:
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(""))
        self.workspace.reference_generation_prompt_edit.setText(
            "Latest first-object draft"
        )
        self.workspace.generate_all_references_button.click()
        self.assertEqual(len(batch_fixtures._ControlledReferenceThread.created), 3)
        self.assertEqual(
            batch_fixtures._ControlledReferenceThread.created[0].prompt,
            "Latest first-object draft",
        )


if __name__ == "__main__":
    unittest.main()
