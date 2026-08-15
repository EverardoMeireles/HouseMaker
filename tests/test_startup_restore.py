# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_state import GenerationData
from housemaker.main import (
    LAST_PROJECT_PATH_SETTING_KEY,
    BlueprintWorkspace,
    MainWindow,
)
from housemaker.models import GROUND_LEVEL_INDEX, create_default_levels
from housemaker.project_io import ProjectData


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _project_data(level_name: str = "Restored project") -> ProjectData:
    levels = create_default_levels()
    levels[GROUND_LEVEL_INDEX].name = level_name
    return ProjectData(
        blueprint_path=None,
        current_level_index=GROUND_LEVEL_INDEX,
        levels=levels,
        generation=GenerationData(),
    )


# ### Startup restoration tests ###
class StartupRestorationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.settings_path = (
            Path(self._temporary_directory.name) / "settings.json"
        )
        self.store = ApplicationSettingsStore(self.settings_path)
        self._widgets: list[MainWindow | BlueprintWorkspace] = []

    def tearDown(self) -> None:
        for widget in reversed(self._widgets):
            if isinstance(widget, MainWindow):
                widget.close()
            else:
                widget.generation.shutdown()
                widget.close()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _track(self, widget):
        self._widgets.append(widget)
        return widget

    def test_main_window_opens_blueprint_workspace_directly(self) -> None:
        with patch("housemaker.main.load_project") as load_project_mock:
            window = self._track(MainWindow(application_settings=self.store))

        self.assertIs(window.centralWidget(), window.blueprint_workspace)
        self.assertFalse(hasattr(window, "home_page"))
        self.assertFalse(hasattr(window, "stack"))
        load_project_mock.assert_not_called()

    def test_main_window_restores_last_successful_project(self) -> None:
        project_path = Path(self._temporary_directory.name) / "restored.json"
        normalized_path = str(project_path.resolve())
        self.assertTrue(
            self.store.set(LAST_PROJECT_PATH_SETTING_KEY, normalized_path)
        )

        with patch(
            "housemaker.main.load_project",
            return_value=_project_data(),
        ) as load_project_mock:
            window = self._track(MainWindow(application_settings=self.store))

        load_project_mock.assert_called_once_with(normalized_path)
        self.assertEqual(
            window.blueprint_workspace.current_level.name,
            "Restored project",
        )
        self.assertEqual(
            window.blueprint_workspace.current_project_path,
            normalized_path,
        )

    def test_missing_or_corrupt_last_project_falls_back_and_clears_path(
        self,
    ) -> None:
        missing_path = Path(self._temporary_directory.name) / "missing.json"
        corrupt_path = Path(self._temporary_directory.name) / "corrupt.json"
        corrupt_path.write_text("[not an object]", encoding="utf-8")

        for project_path in (missing_path, corrupt_path):
            with self.subTest(path=project_path.name):
                case_settings_path = (
                    Path(self._temporary_directory.name)
                    / f"{project_path.stem}-settings.json"
                )
                case_store = ApplicationSettingsStore(case_settings_path)
                case_store.set(
                    LAST_PROJECT_PATH_SETTING_KEY,
                    str(project_path.resolve()),
                )

                window = self._track(
                    MainWindow(application_settings=case_store)
                )

                self.assertEqual(
                    window.blueprint_workspace.current_level.name,
                    create_default_levels()[GROUND_LEVEL_INDEX].name,
                )
                self.assertIsNone(
                    case_store.get(LAST_PROJECT_PATH_SETTING_KEY)
                )

    def test_successful_save_and_load_update_last_project_path(self) -> None:
        workspace = self._track(
            BlueprintWorkspace(application_settings=self.store)
        )
        save_path = Path(self._temporary_directory.name) / "saved.json"
        load_path = Path(self._temporary_directory.name) / "loaded.json"

        with (
            patch(
                "housemaker.main.QFileDialog.getSaveFileName",
                return_value=(str(save_path), "JSON Files (*.json)"),
            ),
            patch("housemaker.main.save_project"),
            patch("housemaker.main.QMessageBox.information"),
        ):
            workspace._handle_save_clicked()

        self.assertEqual(
            self.store.get(LAST_PROJECT_PATH_SETTING_KEY),
            str(save_path.resolve()),
        )

        with (
            patch(
                "housemaker.main.QFileDialog.getOpenFileName",
                return_value=(str(load_path), "JSON Files (*.json)"),
            ),
            patch(
                "housemaker.main.load_project",
                return_value=_project_data("Manually loaded project"),
            ),
        ):
            workspace._handle_load_clicked()

        self.assertEqual(workspace.current_level.name, "Manually loaded project")
        self.assertEqual(
            self.store.get(LAST_PROJECT_PATH_SETTING_KEY),
            str(load_path.resolve()),
        )

    def test_failed_load_does_not_replace_last_successful_path(self) -> None:
        workspace = self._track(
            BlueprintWorkspace(application_settings=self.store)
        )
        previous_path = str(
            (Path(self._temporary_directory.name) / "previous.json").resolve()
        )
        failed_path = str(
            (Path(self._temporary_directory.name) / "failed.json").resolve()
        )
        self.store.set(LAST_PROJECT_PATH_SETTING_KEY, previous_path)

        with (
            patch(
                "housemaker.main.QFileDialog.getOpenFileName",
                return_value=(failed_path, "JSON Files (*.json)"),
            ),
            patch(
                "housemaker.main.load_project",
                side_effect=ValueError("invalid project"),
            ),
            patch("housemaker.main.QMessageBox.critical"),
        ):
            workspace._handle_load_clicked()

        self.assertEqual(
            self.store.get(LAST_PROJECT_PATH_SETTING_KEY),
            previous_path,
        )


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
