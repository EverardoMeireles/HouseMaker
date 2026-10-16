# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path

from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.ceiling_height_dialog import (
    CEILING_HEIGHT_INTERVAL_SETTING_KEY,
    CEILING_HEIGHT_MODEL_SETTING_KEY,
    CeilingHeightAnalysisDialog,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Dialog tests ###
class CeilingHeightAnalysisDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.settings = ApplicationSettingsStore(
            Path(self.temporary_directory.name) / "settings.json"
        )
        self.dialog = CeilingHeightAnalysisDialog(
            application_settings=self.settings
        )

    def tearDown(self) -> None:
        self.dialog.dispose()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_whole_video_analysis_emits_model_interval_and_starting_level(
        self,
    ) -> None:
        analysis_requested = QSignalSpy(self.dialog.analysis_requested)

        self.dialog.set_video_metadata(120, fps=24.0)
        self.dialog.set_level_options(
            (("Underground 1", 0), ("Ground", 1), ("Story 1", 2)),
            current_level_index=1,
        )
        self.dialog.analysis_button.click()

        self.assertFalse(hasattr(self.dialog, "mark_start_button"))
        self.assertFalse(hasattr(self.dialog, "mark_end_button"))
        self.assertFalse(hasattr(self.dialog, "marked_frame_range"))
        self.assertEqual(self.dialog.frame_analysis_interval, 15)
        self.assertEqual(self.dialog.selected_level_index, 1)
        self.assertEqual(analysis_requested.count(), 1)
        self.assertEqual(
            list(analysis_requested.at(0)),
            ["gpt-5.6-luna", 15, 1],
        )

    def test_running_analysis_changes_button_to_cancel(self) -> None:
        cancellation_requested = QSignalSpy(
            self.dialog.cancellation_requested
        )
        analysis_requested = QSignalSpy(self.dialog.analysis_requested)
        self.dialog.set_video_metadata(60, fps=30.0)
        self.dialog.set_level_options((("Ground", 1),), current_level_index=1)

        self.dialog.set_running(True)
        self.dialog.set_progress(2, 4, current_frame=30)
        self.dialog.analysis_button.click()

        self.assertEqual(cancellation_requested.count(), 1)
        self.assertEqual(analysis_requested.count(), 0)
        self.assertEqual(self.dialog.progress_bar.maximum(), 4)
        self.assertEqual(self.dialog.progress_bar.value(), 2)
        self.assertIn("video frame 31", self.dialog.status_label.text())
        self.assertFalse(self.dialog.progress_bar.isHidden())

        self.dialog.set_running(False)
        self.assertTrue(self.dialog.progress_bar.isHidden())

    def test_level_options_select_the_requested_current_level(self) -> None:
        self.dialog.set_level_options(
            (("Underground 1", 0), ("Ground", 1), ("Story 1", 2)),
            current_level_index=2,
        )

        self.assertEqual(self.dialog.current_level_combo.count(), 3)
        self.assertEqual(self.dialog.current_level_combo.currentText(), "Story 1")
        self.assertEqual(self.dialog.selected_level_index, 2)

        self.dialog.current_level_combo.setCurrentIndex(0)
        self.assertEqual(self.dialog.selected_level_index, 0)

    def test_paid_analysis_button_shows_dynamic_cost_but_local_qwen_does_not(
        self,
    ) -> None:
        self.dialog.set_level_options((("Ground", 1),), current_level_index=1)
        self.dialog.set_video_metadata(
            350,
            fps=30.0,
            width=1_920,
            height=1_080,
        )

        self.dialog.frame_interval_spinbox.setValue(15)
        fifteen_frame_text = self.dialog.analysis_button.text()
        self.assertRegex(fifteen_frame_text, r"\([^)]*\$\d+\.\d{2}[^)]*\)$")

        self.dialog.frame_interval_spinbox.setValue(30)
        thirty_frame_text = self.dialog.analysis_button.text()
        self.assertRegex(thirty_frame_text, r"\([^)]*\$\d+\.\d{2}[^)]*\)$")
        self.assertNotEqual(fifteen_frame_text, thirty_frame_text)

        local_index = self.dialog.model_combo.findData("qwen3-vl-8b-local")
        self.assertGreaterEqual(local_index, 0)
        self.dialog.model_combo.setCurrentIndex(local_index)
        self.assertNotIn("$", self.dialog.analysis_button.text())

    def test_model_and_interval_are_persisted(self) -> None:
        terra_index = self.dialog.model_combo.findData("gpt-5.6-terra")
        self.assertGreaterEqual(terra_index, 0)
        self.dialog.model_combo.setCurrentIndex(terra_index)
        self.dialog.frame_interval_spinbox.setValue(45)

        restored = CeilingHeightAnalysisDialog(
            application_settings=self.settings
        )
        try:
            self.assertEqual(restored.selected_model, "gpt-5.6-terra")
            self.assertEqual(restored.frame_analysis_interval, 45)
            self.assertEqual(
                self.settings.get(CEILING_HEIGHT_MODEL_SETTING_KEY),
                "gpt-5.6-terra",
            )
            self.assertEqual(
                self.settings.get(CEILING_HEIGHT_INTERVAL_SETTING_KEY),
                45,
            )
        finally:
            restored.dispose()
            _qt_application.processEvents()

    def test_close_hides_without_destroying_dialog(self) -> None:
        self.dialog.show()
        _qt_application.processEvents()
        self.assertTrue(self.dialog.isVisible())

        self.dialog.close()
        _qt_application.processEvents()

        self.assertFalse(self.dialog.isVisible())
        self.dialog.show()
        _qt_application.processEvents()
        self.assertTrue(self.dialog.isVisible())


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
