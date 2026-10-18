# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QHideEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFormLayout

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.reference_generation_models import (
    DEFAULT_REFERENCE_GENERATION_MODEL,
    REFERENCE_GENERATION_MODEL_OPTIONS,
    REFERENCE_GENERATION_MODEL_SETTINGS_KEY,
    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
)
from housemaker.reference_view_presets import (
    DEFAULT_REFERENCE_VIEW_PRESETS,
    REFERENCE_VIEW_PRESETS_SETTINGS_KEY,
    ReferenceViewPreset,
    read_reference_view_presets,
    save_reference_view_presets,
)
from housemaker.settings_widget import SettingsWidget, read_reference_generation_model

# ### Qt application ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Widget cleanup ###
def _dispose_test_settings(widget: SettingsWidget) -> None:
    widget.dispose()
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    _qt_application.processEvents()


# ### Settings editor tests ###
class ReferenceViewPresetsSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.store = ApplicationSettingsStore(
            Path(temporary_directory.name) / "settings.json"
        )
        self.widget = self._build_widget()

    def _build_widget(self) -> SettingsWidget:
        widget = SettingsWidget(application_settings=self.store, environment={})
        self.addCleanup(_dispose_test_settings, widget)
        return widget

    def test_reference_model_selector_is_above_views_and_lists_all_providers(self) -> None:
        combo = self.widget.reference_generation_model_combo
        group = self.widget.reference_view_presets_group
        self.assertTrue(group.isAncestorOf(combo))
        self.assertEqual(
            tuple(
                (combo.itemText(index), combo.itemData(index))
                for index in range(combo.count())
            ),
            REFERENCE_GENERATION_MODEL_OPTIONS,
        )
        self.assertEqual(combo.currentData(), DEFAULT_REFERENCE_GENERATION_MODEL)
        self.assertEqual(
            self.widget.get_settings().reference_generation_model,
            DEFAULT_REFERENCE_GENERATION_MODEL,
        )
        model_form = group.layout().itemAt(0).layout()
        self.assertIsInstance(model_form, QFormLayout)
        assert isinstance(model_form, QFormLayout)
        self.assertEqual(
            model_form.labelForField(combo).text(), "Reference generation model"
        )
        self.assertIs(
            group.layout().itemAt(1).widget(), self.widget.reference_view_preset_list
        )

    def test_reference_model_selection_persists_using_the_existing_key(self) -> None:
        changes: list[bool] = []
        view_changes: list[object] = []
        self.widget.settings_changed.connect(lambda: changes.append(True))
        self.widget.reference_view_presets_changed.connect(view_changes.append)
        combo = self.widget.reference_generation_model_combo

        combo.setCurrentIndex(
            combo.findData(REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION)
        )

        self.assertEqual(
            self.store.get(REFERENCE_GENERATION_MODEL_SETTINGS_KEY),
            REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
        )
        self.assertEqual(
            self.widget.get_settings().reference_generation_model,
            REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
        )
        self.assertEqual(changes, [True])
        self.assertEqual(view_changes, [])
        restored = self._build_widget()
        self.assertEqual(
            restored.reference_generation_model_combo.currentData(),
            REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
        )

    def test_previous_generation_model_preference_is_loaded_from_disk(self) -> None:
        self.store.set(
            REFERENCE_GENERATION_MODEL_SETTINGS_KEY,
            REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
        )
        restored = SettingsWidget(
            application_settings=ApplicationSettingsStore(self.store.path),
            environment={},
        )
        self.addCleanup(_dispose_test_settings, restored)

        self.assertEqual(
            restored.reference_generation_model_combo.currentData(),
            REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
        )

    def test_invalid_reference_model_preference_falls_back_to_the_default(self) -> None:
        for invalid_model in ("removed-provider", 7, None):
            with self.subTest(model=invalid_model):
                self.store.set(REFERENCE_GENERATION_MODEL_SETTINGS_KEY, invalid_model)
                self.assertEqual(
                    read_reference_generation_model(self.store),
                    DEFAULT_REFERENCE_GENERATION_MODEL,
                )
                restored = self._build_widget()
                self.assertEqual(
                    restored.reference_generation_model_combo.currentData(),
                    DEFAULT_REFERENCE_GENERATION_MODEL,
                )

    def test_six_defaults_load_in_an_editable_settings_section(self) -> None:
        self.assertEqual(
            self.widget.get_reference_view_presets(),
            DEFAULT_REFERENCE_VIEW_PRESETS,
        )
        self.assertEqual(self.widget.reference_view_preset_list.count(), 6)
        self.assertEqual(
            self.widget.reference_view_name_edit.text(),
            DEFAULT_REFERENCE_VIEW_PRESETS[0].name,
        )
        self.assertEqual(
            self.widget.reference_view_prompt_edit.toPlainText(),
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
        )
        self.assertEqual(
            self.widget.reference_view_presets_group.title(),
            "Reference generation views",
        )
        for control in (
            self.widget.reference_view_preset_list,
            self.widget.reference_view_name_edit,
            self.widget.reference_view_prompt_edit,
            self.widget.add_reference_view_button,
            self.widget.remove_reference_view_button,
        ):
            self.assertTrue(
                self.widget.reference_view_presets_group.isAncestorOf(control)
            )
        self.assertIsNone(self.store.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY))

    def test_add_and_remove_views_emit_only_the_dedicated_signal(self) -> None:
        changes: list[object] = []
        ordinary_changes: list[bool] = []
        self.widget.reference_view_presets_changed.connect(changes.append)
        self.widget.settings_changed.connect(lambda: ordinary_changes.append(True))

        self.widget.add_reference_view_button.click()

        presets = read_reference_view_presets(self.store)
        self.assertEqual(len(presets), 7)
        self.assertEqual(presets[-1].name, "New view 1")
        self.assertEqual(self.widget.reference_view_preset_list.currentRow(), 6)
        self.assertTrue(self.widget.reference_view_prompt_edit.toPlainText())
        self.assertEqual(changes, [presets])
        self.assertEqual(ordinary_changes, [])

        self.widget.remove_reference_view_button.click()

        self.assertEqual(
            read_reference_view_presets(self.store), DEFAULT_REFERENCE_VIEW_PRESETS
        )
        self.assertEqual(changes[-1], DEFAULT_REFERENCE_VIEW_PRESETS)
        self.assertEqual(len(changes), 2)
        self.assertEqual(ordinary_changes, [])

    def test_empty_list_remains_empty_when_reopened(self) -> None:
        while self.widget.reference_view_preset_list.count():
            self.widget.remove_reference_view_button.click()

        self.assertEqual(self.store.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY), [])
        self.assertEqual(self.widget.get_reference_view_presets(), ())
        self.assertFalse(self.widget.reference_view_name_edit.isEnabled())
        self.assertFalse(self.widget.reference_view_prompt_edit.isEnabled())
        self.assertFalse(self.widget.remove_reference_view_button.isEnabled())
        restored = self._build_widget()
        self.assertEqual(restored.reference_view_preset_list.count(), 0)
        restored.add_reference_view_button.click()
        self.assertEqual(restored.reference_view_preset_list.count(), 1)
        self.assertEqual(read_reference_view_presets(self.store)[0].name, "New view 1")

    def test_name_and_multiline_prompt_save_without_resetting_selection_or_cursor(
        self,
    ) -> None:
        self.widget.reference_view_preset_list.setCurrentRow(2)
        self.widget.reference_view_name_edit.setText("My elevated view")
        self.widget.reference_view_prompt_edit.setPlainText(
            "Show the object from above.\nKeep all its original details."
        )
        cursor = self.widget.reference_view_prompt_edit.textCursor()
        cursor.setPosition(12)
        self.widget.reference_view_prompt_edit.setTextCursor(cursor)

        self.widget.save_reference_view_button.click()

        saved = read_reference_view_presets(self.store)[2]
        self.assertEqual(saved.name, "My elevated view")
        self.assertEqual(
            saved.prompt,
            "Show the object from above.\nKeep all its original details.",
        )
        self.assertEqual(self.widget.reference_view_preset_list.currentRow(), 2)
        self.assertEqual(
            self.widget.reference_view_preset_list.item(2).text(),
            "My elevated view",
        )
        self.assertEqual(
            self.widget.reference_view_prompt_edit.textCursor().position(),
            12,
        )
        self.assertFalse(self.widget.save_reference_view_button.isEnabled())

    def test_valid_edits_autosave_after_a_short_debounce(self) -> None:
        self.widget.reference_view_name_edit.setText("Front product view")
        self.assertIsNone(self.store.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY))

        QTest.qWait(450)

        self.assertEqual(
            read_reference_view_presets(self.store)[0].name,
            "Front product view",
        )
        self.assertFalse(self.widget.save_reference_view_button.isEnabled())

    def test_switching_views_flushes_valid_edits_to_the_correct_view(self) -> None:
        first_id = DEFAULT_REFERENCE_VIEW_PRESETS[0].preset_id
        self.widget.reference_view_prompt_edit.setPlainText("A custom frontal view.")

        self.widget.reference_view_preset_list.setCurrentRow(3)

        self.assertEqual(
            read_reference_view_presets(self.store)[0],
            ReferenceViewPreset(
                first_id,
                DEFAULT_REFERENCE_VIEW_PRESETS[0].name,
                "A custom frontal view.",
            ),
        )
        self.assertEqual(
            self.widget.reference_view_prompt_edit.toPlainText(),
            DEFAULT_REFERENCE_VIEW_PRESETS[3].prompt,
        )

    def test_leaving_settings_flushes_valid_edits_before_generation_opens(
        self,
    ) -> None:
        self.widget.reference_view_name_edit.setText("Immediately saved view")

        self.widget.hideEvent(QHideEvent())

        self.assertEqual(
            read_reference_view_presets(self.store)[0].name,
            "Immediately saved view",
        )

    def test_blank_and_oversized_drafts_do_not_replace_saved_values(self) -> None:
        self.assertTrue(
            save_reference_view_presets(self.store, DEFAULT_REFERENCE_VIEW_PRESETS)
        )
        for value in ("", "   ", "x" * 4001):
            with self.subTest(prompt_length=len(value)):
                self.widget.reference_view_prompt_edit.setPlainText(value)
                self.widget._save_reference_view_edits()
                self.assertEqual(
                    read_reference_view_presets(self.store),
                    DEFAULT_REFERENCE_VIEW_PRESETS,
                )
                self.assertFalse(self.widget.save_reference_view_button.isEnabled())

        self.widget.reference_view_prompt_edit.setPlainText(
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt
        )
        self.widget.reference_view_name_edit.clear()
        self.widget._save_reference_view_edits()
        self.assertEqual(
            read_reference_view_presets(self.store), DEFAULT_REFERENCE_VIEW_PRESETS
        )

    def test_duplicate_view_name_is_rejected_without_writing(self) -> None:
        self.widget.reference_view_name_edit.setText(
            DEFAULT_REFERENCE_VIEW_PRESETS[1].name.upper()
        )
        self.widget._save_reference_view_edits()

        self.assertIsNone(self.store.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY))
        self.assertEqual(
            self.widget.get_reference_view_presets(), DEFAULT_REFERENCE_VIEW_PRESETS
        )
        self.assertIn(
            "unique",
            self.widget.reference_view_preset_status_label.text(),
        )

    def test_custom_prompt_option_name_is_reserved(self) -> None:
        for name in ("(Write prompt)", " (WRITE PROMPT) "):
            with self.subTest(name=name):
                self.widget.reference_view_name_edit.setText(name)
                self.widget._save_reference_view_edits()
                self.assertIsNone(self.store.get(REFERENCE_VIEW_PRESETS_SETTINGS_KEY))
                self.assertEqual(
                    self.widget.get_reference_view_presets(),
                    DEFAULT_REFERENCE_VIEW_PRESETS,
                )
                self.assertFalse(self.widget.save_reference_view_button.isEnabled())
                self.assertIn(
                    "reserved",
                    self.widget.reference_view_preset_status_label.text(),
                )

    def test_failed_write_preserves_saved_state_and_can_be_retried(self) -> None:
        changes: list[object] = []
        self.widget.reference_view_presets_changed.connect(changes.append)
        self.widget.reference_view_name_edit.setText("Retry this view")

        with patch.object(self.store, "set", return_value=False):
            self.widget.save_reference_view_button.click()

        self.assertEqual(
            self.widget.get_reference_view_presets(), DEFAULT_REFERENCE_VIEW_PRESETS
        )
        self.assertEqual(
            self.widget.reference_view_preset_list.item(0).text(),
            DEFAULT_REFERENCE_VIEW_PRESETS[0].name,
        )
        self.assertEqual(self.widget.reference_view_name_edit.text(), "Retry this view")
        self.assertEqual(changes, [])
        self.assertIn(
            "Could not save", self.widget.reference_view_preset_status_label.text()
        )
        self.assertTrue(self.widget.save_reference_view_button.isEnabled())

        self.widget.save_reference_view_button.click()

        self.assertEqual(
            read_reference_view_presets(self.store)[0].name, "Retry this view"
        )
        self.assertEqual(len(changes), 1)

    def test_failed_write_keeps_draft_selected_when_switching_views(self) -> None:
        prompt = "A new prompt that must not be lost."
        self.widget.reference_view_prompt_edit.setPlainText(prompt)

        with patch.object(self.store, "set", return_value=False):
            self.widget.reference_view_preset_list.setCurrentRow(3)

        self.assertEqual(self.widget.reference_view_preset_list.currentRow(), 0)
        self.assertEqual(self.widget.reference_view_prompt_edit.toPlainText(), prompt)
        self.assertIn(
            "Could not save", self.widget.reference_view_preset_status_label.text()
        )

        self.widget.reference_view_preset_list.setCurrentRow(3)

        self.assertEqual(self.widget.reference_view_preset_list.currentRow(), 3)
        self.assertEqual(read_reference_view_presets(self.store)[0].prompt, prompt)

    def test_failed_draft_save_prevents_add_from_discarding_it(self) -> None:
        self.widget.reference_view_name_edit.setText("Keep this draft")

        with patch.object(self.store, "set", return_value=False) as write:
            self.widget.add_reference_view_button.click()

        self.assertEqual(write.call_count, 1)
        self.assertEqual(self.widget.reference_view_preset_list.count(), 6)
        self.assertEqual(self.widget.reference_view_name_edit.text(), "Keep this draft")
        self.assertIn(
            "Could not save", self.widget.reference_view_preset_status_label.text()
        )

        self.widget.add_reference_view_button.click()

        presets = read_reference_view_presets(self.store)
        self.assertEqual(presets[0].name, "Keep this draft")
        self.assertEqual(len(presets), 7)

    def test_failed_add_or_remove_does_not_change_the_visible_list(self) -> None:
        with patch.object(self.store, "set", return_value=False):
            self.widget.add_reference_view_button.click()
            self.assertEqual(self.widget.reference_view_preset_list.count(), 6)
            self.widget.remove_reference_view_button.click()
            self.assertEqual(self.widget.reference_view_preset_list.count(), 6)
        self.assertEqual(
            self.widget.get_reference_view_presets(), DEFAULT_REFERENCE_VIEW_PRESETS
        )

    def test_new_view_names_do_not_duplicate_existing_names(self) -> None:
        self.widget.add_reference_view_button.click()
        self.widget.add_reference_view_button.click()

        presets = read_reference_view_presets(self.store)
        self.assertEqual(
            [item.name for item in presets[-2:]], ["New view 1", "New view 2"]
        )
        self.assertNotEqual(presets[-1].preset_id, presets[-2].preset_id)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
