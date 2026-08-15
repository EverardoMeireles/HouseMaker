# ### Environment setup ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QLineEdit

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.settings_widget import (
    CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY,
    DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
    FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY,
    MESHY_API_KEY_ENVIRONMENT_VARIABLE,
    MESHY_API_KEY_SETTING_KEY,
    OPENAI_API_KEY_ENVIRONMENT_VARIABLE,
    OPENAI_API_KEY_SETTING_KEY,
    Fullscreen3DViewerScreenOption,
    SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
    SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
    GenerationServiceSettings,
    SettingsWidget,
    fullscreen_3d_viewer_screen_id,
    read_canvas_3d_navigation_toggle_hotkey,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test cases ###
class SettingsWidgetTests(unittest.TestCase):
    def test_api_key_fields_and_fullscreen_display_selector_are_visible(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            widget = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )

            self.assertTrue(hasattr(widget, "meshy_api_key_edit"))
            self.assertTrue(hasattr(widget, "openai_api_key_edit"))
            self.assertTrue(
                hasattr(widget, "fullscreen_3d_viewer_screen_combo")
            )
            self.assertTrue(
                hasattr(
                    widget,
                    "canvas_3d_navigation_toggle_hotkey_edit",
                )
            )
            self.assertFalse(hasattr(widget, "surface_texture_provider_combo"))

    def test_meshy_api_key_persists_in_application_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=settings,
                environment={},
            )
            widget.meshy_api_key_edit.setText("meshy-secret")

            self.assertEqual(
                settings.get(MESHY_API_KEY_SETTING_KEY),
                "meshy-secret",
            )

            restored_widget = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertEqual(
                restored_widget.get_settings().meshy_api_key,
                "meshy-secret",
            )

    def test_environment_key_is_used_without_copying_it_into_the_input(self) -> None:
        environment = {
            MESHY_API_KEY_ENVIRONMENT_VARIABLE: "meshy-environment",
            OPENAI_API_KEY_ENVIRONMENT_VARIABLE: "openai-environment",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            widget = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment=environment,
            )

            self.assertEqual(widget.meshy_api_key_edit.text(), "")
            self.assertIn(
                MESHY_API_KEY_ENVIRONMENT_VARIABLE,
                widget.meshy_key_status_label.text(),
            )
            self.assertEqual(
                widget.get_settings().meshy_api_key,
                "meshy-environment",
            )
            self.assertEqual(widget.openai_api_key_edit.text(), "")
            self.assertEqual(
                widget.get_settings().openai_api_key,
                "openai-environment",
            )

    def test_saved_key_overrides_environment_and_is_masked(self) -> None:
        environment = {MESHY_API_KEY_ENVIRONMENT_VARIABLE: "meshy-environment"}
        with tempfile.TemporaryDirectory() as temporary_directory:
            widget = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment=environment,
            )
            widget.meshy_api_key_edit.setText("meshy-session")

            settings = widget.get_settings()
            self.assertEqual(settings.meshy_api_key, "meshy-session")
            self.assertEqual(
                widget.meshy_api_key_edit.echoMode(),
                QLineEdit.EchoMode.Password,
            )
            self.assertNotIn("meshy-session", repr(settings))

    def test_openai_key_and_external_surface_provider_are_read_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            widget.openai_api_key_edit.setText("openai-secret")
            application_settings.set(
                SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
                SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
            )

            self.assertEqual(
                application_settings.get(OPENAI_API_KEY_SETTING_KEY),
                "openai-secret",
            )
            self.assertEqual(
                application_settings.get(SURFACE_TEXTURE_PROVIDER_SETTING_KEY),
                SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
            )
            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            restored_settings = restored.get_settings()
            self.assertEqual(restored_settings.openai_api_key, "openai-secret")
            self.assertEqual(
                restored_settings.surface_texture_provider,
                SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
            )
            self.assertNotIn("openai-secret", repr(restored_settings))

    def test_settings_model_rejects_invalid_smart_topology_polycounts(self) -> None:
        for value in (99, 15_001, True, 4_000.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "target polycount"):
                    GenerationServiceSettings(
                        meshy_api_key="",
                        meshy_target_polycount=value,  # type: ignore[arg-type]
                    )

    def test_canvas_3d_navigation_hotkey_uses_safe_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )

            self.assertEqual(
                widget.get_settings().canvas_3d_navigation_toggle_hotkey,
                DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
            )
            self.assertEqual(
                read_canvas_3d_navigation_toggle_hotkey(
                    application_settings
                ),
                DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
            )
            self.assertEqual(
                widget.canvas_3d_navigation_toggle_hotkey_edit
                .keySequence()
                .toString(QKeySequence.SequenceFormat.PortableText),
                DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
            )

    def test_canvas_3d_navigation_hotkey_persists_and_emits_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )

            widget.canvas_3d_navigation_toggle_hotkey_edit.setKeySequence(
                QKeySequence(
                    "Ctrl+Alt+Delete",
                    QKeySequence.SequenceFormat.PortableText,
                )
            )

            self.assertEqual(
                application_settings.get(
                    CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY
                ),
                "Ctrl+Alt+Del",
            )
            self.assertEqual(
                widget.get_settings().canvas_3d_navigation_toggle_hotkey,
                "Ctrl+Alt+Del",
            )
            self.assertEqual(len(emitted_changes), 1)

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertEqual(
                restored.get_settings().canvas_3d_navigation_toggle_hotkey,
                "Ctrl+Alt+Del",
            )

    def test_canvas_3d_navigation_hotkey_rejects_invalid_sequences(self) -> None:
        invalid_hotkeys: tuple[object, ...] = (
            None,
            "",
            "garbage",
            "Ctrl+Shift",
            "Z",
            "Q",
            "S",
            "D",
            "R",
            "F",
            "Z, Q",
        )
        for hotkey in invalid_hotkeys:
            with self.subTest(hotkey=hotkey):
                with self.assertRaisesRegex(ValueError, "navigation toggle"):
                    GenerationServiceSettings(
                        canvas_3d_navigation_toggle_hotkey=hotkey,  # type: ignore[arg-type]
                    )

        for hotkey in (
            "Ctrl+Z",
            "Alt+Q",
            "Shift+S",
            "Meta+D",
            "Ctrl+R",
            "Alt+F",
        ):
            with self.subTest(hotkey=hotkey):
                self.assertEqual(
                    GenerationServiceSettings(
                        canvas_3d_navigation_toggle_hotkey=hotkey
                    ).canvas_3d_navigation_toggle_hotkey,
                    hotkey,
                )

    def test_blocked_movement_hotkey_resets_editor_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )

            widget.canvas_3d_navigation_toggle_hotkey_edit.setKeySequence(
                QKeySequence("Q", QKeySequence.SequenceFormat.PortableText)
            )

            self.assertEqual(
                application_settings.get(
                    CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY
                ),
                DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
            )
            self.assertEqual(
                widget.get_settings().canvas_3d_navigation_toggle_hotkey,
                DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
            )

    def test_invalid_saved_canvas_3d_navigation_hotkey_falls_back_to_default(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            application_settings.set(
                CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY,
                "not-a-hotkey",
            )

            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )

            self.assertEqual(
                read_canvas_3d_navigation_toggle_hotkey(
                    application_settings
                ),
                DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
            )
            self.assertEqual(
                widget.get_settings().canvas_3d_navigation_toggle_hotkey,
                DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
            )

    def test_legacy_f_toggle_hotkey_falls_back_to_safe_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            application_settings.set(
                CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY,
                "F",
            )

            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )

            self.assertEqual(
                widget.get_settings().canvas_3d_navigation_toggle_hotkey,
                DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
            )
            self.assertEqual(
                read_canvas_3d_navigation_toggle_hotkey(
                    application_settings
                ),
                DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
            )

    def test_fullscreen_3d_viewer_display_persists_selected_screen_id(self) -> None:
        options = (
            Fullscreen3DViewerScreenOption(
                screen_id="monitor:Acme|Panel|001",
                label="Main display (2560 × 1440)",
            ),
            Fullscreen3DViewerScreenOption(
                screen_id="monitor:Acme|Panel|002",
                label="Second display (1920 × 1080)",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            with patch(
                "housemaker.settings_widget."
                "connected_fullscreen_3d_viewer_display_options",
                return_value=options,
            ):
                widget = SettingsWidget(
                    application_settings=application_settings,
                    environment={},
                )
                combo = widget.fullscreen_3d_viewer_screen_combo

                self.assertEqual(combo.itemText(0), "None")
                self.assertIsNone(combo.itemData(0))
                self.assertEqual(
                    [combo.itemData(index) for index in range(combo.count())],
                    [None, options[0].screen_id, options[1].screen_id],
                )

                combo.setCurrentIndex(2)

                self.assertEqual(
                    application_settings.get(
                        FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY
                    ),
                    options[1].screen_id,
                )
                self.assertEqual(
                    widget.get_settings().fullscreen_3d_viewer_screen_id,
                    options[1].screen_id,
                )

                restored = SettingsWidget(
                    application_settings=_build_test_settings(
                        temporary_directory
                    ),
                    environment={},
                )
                self.assertEqual(
                    restored.fullscreen_3d_viewer_screen_combo.currentData(),
                    options[1].screen_id,
                )

    def test_fullscreen_3d_viewer_screen_id_prefers_monitor_serial(self) -> None:
        serial_screen = _FakeScreen(
            name="\\\\.\\DISPLAY2",
            manufacturer="Acme",
            model="Studio Panel",
            serial_number="A1B2C3",
        )
        name_only_screen = _FakeScreen(name="HDMI-1")

        self.assertEqual(
            fullscreen_3d_viewer_screen_id(serial_screen),
            "monitor:Acme|Studio Panel|A1B2C3",
        )
        self.assertEqual(
            fullscreen_3d_viewer_screen_id(name_only_screen),
            "screen:HDMI-1",
        )

    def test_missing_saved_fullscreen_display_safely_uses_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            application_settings.set(
                FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY,
                "monitor:Missing|Panel|123",
            )
            with patch(
                "housemaker.settings_widget."
                "connected_fullscreen_3d_viewer_display_options",
                return_value=(),
            ):
                widget = SettingsWidget(
                    application_settings=application_settings,
                    environment={},
                )

            self.assertIsNone(
                widget.fullscreen_3d_viewer_screen_combo.currentData()
            )
            self.assertIsNone(
                widget.get_settings().fullscreen_3d_viewer_screen_id
            )
            self.assertEqual(
                application_settings.get(
                    FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY
                ),
                "monitor:Missing|Panel|123",
            )


# ### Test helpers ###
def _build_test_settings(directory: str) -> ApplicationSettingsStore:
    return ApplicationSettingsStore(Path(directory) / "settings.json")


class _FakeScreen:
    def __init__(
        self,
        *,
        name: str,
        manufacturer: str = "",
        model: str = "",
        serial_number: str = "",
    ) -> None:
        self._name = name
        self._manufacturer = manufacturer
        self._model = model
        self._serial_number = serial_number

    def name(self) -> str:
        return self._name

    def manufacturer(self) -> str:
        return self._manufacturer

    def model(self) -> str:
        return self._model

    def serialNumber(self) -> str:
        return self._serial_number


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
