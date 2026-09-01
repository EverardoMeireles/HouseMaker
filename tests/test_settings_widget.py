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
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.settings_widget import (
    CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY,
    DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
    DEFAULT_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
    DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE,
    DEFAULT_MESHY_TARGET_POLYCOUNT,
    DOORWAY_MESH_UPDATE_DELAY_SECONDS_SETTING_KEY,
    DOORWAY_MESH_UPDATE_DELAY_STEP_SECONDS,
    FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY,
    MAXIMUM_FACE_VISIBILITY_PERCENTAGE,
    MAX_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
    MESHY_API_KEY_ENVIRONMENT_VARIABLE,
    MESHY_API_KEY_SETTING_KEY,
    MINIMUM_FACE_VISIBILITY_PERCENTAGE,
    MINIMUM_FACE_VISIBILITY_PERCENTAGE_SETTING_KEY,
    MIN_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
    OPENAI_API_KEY_ENVIRONMENT_VARIABLE,
    OPENAI_API_KEY_SETTING_KEY,
    UNUSED_FACE_REMOVAL_SETTING_KEY,
    USE_UV_RAYCAST_FOR_OBJECT_GENERATION_SETTING_KEY,
    Fullscreen3DViewerScreenOption,
    SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
    SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
    GenerationServiceSettings,
    SettingsWidget,
    fullscreen_3d_viewer_screen_id,
    read_canvas_3d_navigation_toggle_hotkey,
    read_doorway_mesh_update_delay_seconds,
    read_minimum_face_visibility_percentage,
    read_unused_face_removal,
    read_use_uv_raycast_for_object_generation,
)


# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test cases ###
class SettingsWidgetTests(unittest.TestCase):
    def test_generation_service_defaults_to_two_thousand_target_tris(self) -> None:
        self.assertEqual(DEFAULT_MESHY_TARGET_POLYCOUNT, 2_000)
        self.assertEqual(
            GenerationServiceSettings().meshy_target_polycount,
            2_000,
        )

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
            self.assertTrue(
                hasattr(widget, "doorway_mesh_update_delay_spinbox")
            )
            self.assertFalse(hasattr(widget, "surface_texture_provider_combo"))
            self.assertTrue(hasattr(widget, "unused_face_removal_checkbox"))
            self.assertTrue(
                hasattr(
                    widget,
                    "use_uv_raycast_for_object_generation_checkbox",
                )
            )
            self.assertTrue(
                hasattr(
                    widget,
                    "minimum_face_visibility_percentage_spinbox",
                )
            )
            self.assertFalse(
                hasattr(widget, "project_uvs_from_camera_views_checkbox")
            )
            self.assertFalse(
                hasattr(widget, "camera_validated_simplification_checkbox")
            )
            self.assertFalse(
                hasattr(widget, "simplification_pixel_tolerance_spinbox")
            )

    def test_fullscreen_display_id_can_be_read_without_settings_file_io(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            widget.fullscreen_3d_viewer_screen_combo.addItem(
                "Cached display",
                "screen:cached",
            )
            widget.fullscreen_3d_viewer_screen_combo.setCurrentIndex(
                widget.fullscreen_3d_viewer_screen_combo.findData(
                    "screen:cached"
                )
            )

            with patch.object(
                application_settings,
                "get",
                side_effect=AssertionError("unexpected settings-file read"),
            ):
                screen_id = widget.get_fullscreen_3d_viewer_screen_id()

            self.assertEqual(screen_id, "screen:cached")

    def test_unused_face_removal_persists_and_emits_changes(self) -> None:
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

            self.assertFalse(widget.get_settings().unused_face_removal)
            widget.unused_face_removal_checkbox.setChecked(True)

            self.assertTrue(
                application_settings.get(UNUSED_FACE_REMOVAL_SETTING_KEY)
            )
            self.assertTrue(widget.get_settings().unused_face_removal)
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertTrue(restored.get_settings().unused_face_removal)

    def test_legacy_camera_uv_setting_is_ignored_without_ui(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            legacy_setting_key = "generation/project_uvs_from_camera_views"
            application_settings.set(legacy_setting_key, True)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )

            self.assertFalse(
                hasattr(widget, "project_uvs_from_camera_views_checkbox")
            )
            self.assertFalse(
                hasattr(
                    widget.get_settings(),
                    "project_uvs_from_camera_views",
                )
            )
            self.assertNotIn(
                "Project UVs from camera views",
                [label.text() for label in widget.findChildren(QLabel)],
            )
            self.assertTrue(application_settings.get(legacy_setting_key))

    def test_unused_face_removal_rejects_malformed_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unused face removal"):
            GenerationServiceSettings(unused_face_removal=1)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            application_settings.set(UNUSED_FACE_REMOVAL_SETTING_KEY, "yes")

            self.assertFalse(read_unused_face_removal(application_settings))

    def test_weighted_projection_setting_persists_and_emits_changes(self) -> None:
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

            self.assertFalse(
                widget.get_settings().use_uv_raycast_for_object_generation
            )
            self.assertIn(
                "Use weighted camera projection",
                [label.text() for label in widget.findChildren(QLabel)],
            )
            self.assertIn(
                "without island spacing",
                widget.use_uv_raycast_for_object_generation_checkbox.toolTip(),
            )
            widget.use_uv_raycast_for_object_generation_checkbox.setChecked(
                True
            )

            self.assertTrue(
                application_settings.get(
                    USE_UV_RAYCAST_FOR_OBJECT_GENERATION_SETTING_KEY
                )
            )
            self.assertTrue(
                widget.get_settings().use_uv_raycast_for_object_generation
            )
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertTrue(
                restored.get_settings().use_uv_raycast_for_object_generation
            )

    def test_weighted_projection_setting_rejects_malformed_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Weighted camera projection"):
            GenerationServiceSettings(
                use_uv_raycast_for_object_generation=(
                    1  # type: ignore[arg-type]
                )
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            application_settings.set(
                USE_UV_RAYCAST_FOR_OBJECT_GENERATION_SETTING_KEY,
                "yes",
            )

            self.assertFalse(
                read_use_uv_raycast_for_object_generation(
                    application_settings
                )
            )

    def test_uv_raycast_field_preserves_legacy_positional_settings(self) -> None:
        settings = GenerationServiceSettings(
            "meshy-key",
            4_000,
            "openai-key",
            "meshy",
            "viewer-screen",
            "N",
            True,
            "jobs-screen",
            1.25,
        )

        self.assertEqual(settings.jobs_window_screen_id, "jobs-screen")
        self.assertEqual(settings.doorway_mesh_update_delay_seconds, 1.25)
        self.assertFalse(settings.use_uv_raycast_for_object_generation)
        self.assertEqual(
            settings.minimum_face_visibility_percentage,
            DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE,
        )

    def test_minimum_face_visibility_persists_and_emits_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            spinbox = widget.minimum_face_visibility_percentage_spinbox
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )

            self.assertEqual(
                widget.get_settings().minimum_face_visibility_percentage,
                DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE,
            )
            self.assertEqual(
                spinbox.minimum(),
                MINIMUM_FACE_VISIBILITY_PERCENTAGE,
            )
            self.assertEqual(
                spinbox.maximum(),
                MAXIMUM_FACE_VISIBILITY_PERCENTAGE,
            )
            self.assertEqual(spinbox.singleStep(), 1)
            self.assertEqual(spinbox.suffix(), "%")
            self.assertFalse(spinbox.keyboardTracking())
            self.assertIn("tiny glimpses", spinbox.toolTip())

            spinbox.setValue(37)

            self.assertEqual(
                application_settings.get(
                    MINIMUM_FACE_VISIBILITY_PERCENTAGE_SETTING_KEY
                ),
                37,
            )
            self.assertEqual(
                widget.get_settings().minimum_face_visibility_percentage,
                37,
            )
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertEqual(
                restored.get_settings().minimum_face_visibility_percentage,
                37,
            )

    def test_minimum_face_visibility_rejects_malformed_values(self) -> None:
        invalid_values: tuple[object, ...] = (
            True,
            "5",
            5.0,
            MINIMUM_FACE_VISIBILITY_PERCENTAGE - 1,
            MAXIMUM_FACE_VISIBILITY_PERCENTAGE + 1,
        )
        for value in invalid_values:
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Minimum face visibility percentage",
                ):
                    GenerationServiceSettings(
                        minimum_face_visibility_percentage=(
                            value  # type: ignore[arg-type]
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in invalid_values:
                with self.subTest(persisted_value=value):
                    application_settings.set(
                        MINIMUM_FACE_VISIBILITY_PERCENTAGE_SETTING_KEY,
                        value,
                    )
                    self.assertEqual(
                        read_minimum_face_visibility_percentage(
                            application_settings
                        ),
                        DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE,
                    )

    def test_minimum_face_visibility_accepts_range_boundaries(self) -> None:
        for value in (
            MINIMUM_FACE_VISIBILITY_PERCENTAGE,
            MAXIMUM_FACE_VISIBILITY_PERCENTAGE,
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    GenerationServiceSettings(
                        minimum_face_visibility_percentage=value
                    ).minimum_face_visibility_percentage,
                    value,
                )

    def test_doorway_mesh_update_delay_persists_and_emits_changes(self) -> None:
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

            self.assertEqual(
                widget.get_settings().doorway_mesh_update_delay_seconds,
                DEFAULT_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
            )
            self.assertFalse(
                widget.doorway_mesh_update_delay_spinbox.keyboardTracking()
            )
            self.assertEqual(
                widget.doorway_mesh_update_delay_spinbox.minimum(),
                MIN_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
            )
            self.assertEqual(
                widget.doorway_mesh_update_delay_spinbox.maximum(),
                MAX_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
            )
            self.assertEqual(
                widget.doorway_mesh_update_delay_spinbox.singleStep(),
                DOORWAY_MESH_UPDATE_DELAY_STEP_SECONDS,
            )
            self.assertEqual(
                widget.doorway_mesh_update_delay_spinbox.suffix(),
                " s",
            )
            widget.doorway_mesh_update_delay_spinbox.setValue(2.4)

            self.assertEqual(
                application_settings.get(
                    DOORWAY_MESH_UPDATE_DELAY_SECONDS_SETTING_KEY
                ),
                2.4,
            )
            self.assertEqual(
                widget.get_settings().doorway_mesh_update_delay_seconds,
                2.4,
            )
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertEqual(
                restored.get_settings().doorway_mesh_update_delay_seconds,
                2.4,
            )

    def test_doorway_mesh_update_delay_rejects_malformed_values(self) -> None:
        invalid_values: tuple[object, ...] = (
            True,
            "1.0",
            float("nan"),
            float("inf"),
            float("-inf"),
            MIN_DOORWAY_MESH_UPDATE_DELAY_SECONDS - 0.01,
            MAX_DOORWAY_MESH_UPDATE_DELAY_SECONDS + 0.01,
        )
        for value in invalid_values:
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Doorway mesh update delay",
                ):
                    GenerationServiceSettings(
                        doorway_mesh_update_delay_seconds=(
                            value  # type: ignore[arg-type]
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in invalid_values:
                with self.subTest(persisted_value=value):
                    application_settings.set(
                        DOORWAY_MESH_UPDATE_DELAY_SECONDS_SETTING_KEY,
                        value,
                    )
                    self.assertEqual(
                        read_doorway_mesh_update_delay_seconds(
                            application_settings
                        ),
                        DEFAULT_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
                    )

    def test_doorway_mesh_update_delay_accepts_range_boundaries(self) -> None:
        for value in (
            MIN_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
            MAX_DOORWAY_MESH_UPDATE_DELAY_SECONDS,
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    GenerationServiceSettings(
                        doorway_mesh_update_delay_seconds=value
                    ).doorway_mesh_update_delay_seconds,
                    value,
                )
        self.assertIsInstance(
            GenerationServiceSettings(
                doorway_mesh_update_delay_seconds=1
            ).doorway_mesh_update_delay_seconds,
            float,
        )

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
                        canvas_3d_navigation_toggle_hotkey=(
                            hotkey  # type: ignore[arg-type]
                        ),
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
