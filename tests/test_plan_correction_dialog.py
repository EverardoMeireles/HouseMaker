# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from housemaker.plan_correction_dialog import PlanCorrectionDialog

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Dialog tests ###
class PlanCorrectionDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dialog = PlanCorrectionDialog()

    def tearDown(self) -> None:
        self.dialog.close()
        self.dialog.deleteLater()
        _qt_application.processEvents()

    def test_starts_without_a_choice_and_explains_manual_import(self) -> None:
        self.assertIsNone(self.dialog.chosen_action)
        self.assertEqual(self.dialog.windowTitle(), "Image correction")
        self.assertEqual(
            [button.text() for button in self.dialog.findChildren(QPushButton)],
            ["Use api", "Copy prompt", "Browse json"],
        )
        instructions = self.dialog.instructions_label.text()
        self.assertIn("L1, L2, L3", instructions)
        self.assertIn("manually load", instructions)

    def test_api_button_accepts_api_choice(self) -> None:
        self._assert_choice(self.dialog.api_button, "api")

    def test_copy_prompt_button_accepts_copy_choice(self) -> None:
        self._assert_choice(self.dialog.copy_prompt_button, "copy_prompt")

    def test_browse_json_button_accepts_import_choice(self) -> None:
        self._assert_choice(self.dialog.browse_json_button, "browse_json")

    def test_closing_does_not_choose_an_action(self) -> None:
        self.dialog.show()
        self.dialog.close()
        self.assertIsNone(self.dialog.chosen_action)
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Rejected)

    def test_escape_does_not_choose_an_action(self) -> None:
        self.dialog.show()
        QTest.keyClick(self.dialog, Qt.Key.Key_Escape)
        self.assertIsNone(self.dialog.chosen_action)
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Rejected)

    # ### Assertions ###
    def _assert_choice(self, button: QPushButton, expected: str) -> None:
        accepted = QSignalSpy(self.dialog.accepted)
        button.click()
        self.assertEqual(self.dialog.chosen_action, expected)
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(accepted.count(), 1)
