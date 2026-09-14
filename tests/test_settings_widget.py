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
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
)

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.first_person_navigation import (
    DEFAULT_FIRST_PERSON_NAVIGATION_MODE,
    FIRST_PERSON_NAVIGATION_MODE_GRAVITY,
    FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
    FIRST_PERSON_NAVIGATION_MODE_OPTIONS,
)
from housemaker.settings_widget import (
    ATLAS_DISPLAY_SCREEN_SETTING_KEY,
    AUTOMATIC_ATLAS_CREATION_SETTING_KEY,
    AUTOMATIC_ATLAS_TEXTURE_RESOLUTION_SETTING_KEY,
    AUTOMATIC_ATLAS_TEXTURE_RESOLUTIONS,
    AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR_SETTING_KEY,
    CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY_SETTING_KEY,
    DEFAULT_AUTOMATIC_ATLAS_CREATION,
    DEFAULT_AUTOMATIC_ATLAS_TEXTURE_RESOLUTION,
    DEFAULT_AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR,
    DEFAULT_CANVAS_3D_NAVIGATION_TOGGLE_HOTKEY,
    DEFAULT_IGNORE_TOP_DOWN_CEILING,
    DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS,
    DEFAULT_MESHY_TARGET_POLYCOUNT,
    DEFAULT_MINIMUM_FACE_VISIBILITY_PERCENTAGE,
    DEFAULT_SNAP_MIDDLE_EQUAL_ANGLE_ONLY,
    DEFAULT_USE_HALF_MESH_TEXTURE_PREFIX,
    DEFAULT_WALL_VERTEX_UPDATE_DELAY_SECONDS,
    FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY,
    FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY,
    GENERATION_DISPLAY_SCREEN_SETTING_KEY,
    IGNORE_TOP_DOWN_CEILING_SETTING_KEY,
    MAX_MESH_EDIT_UPDATE_DELAY_SECONDS,
    MAX_WALL_VERTEX_UPDATE_DELAY_SECONDS,
    MAXIMUM_FACE_VISIBILITY_PERCENTAGE,
    MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY,
    MESH_EDIT_UPDATE_DELAY_STEP_SECONDS,
    MESHY_API_KEY_ENVIRONMENT_VARIABLE,
    MESHY_API_KEY_SETTING_KEY,
    MIN_MESH_EDIT_UPDATE_DELAY_SECONDS,
    MIN_WALL_VERTEX_UPDATE_DELAY_SECONDS,
    MINIMUM_FACE_VISIBILITY_PERCENTAGE,
    MINIMUM_FACE_VISIBILITY_PERCENTAGE_SETTING_KEY,
    OPENAI_API_KEY_ENVIRONMENT_VARIABLE,
    OPENAI_API_KEY_SETTING_KEY,
    SCENE_3D_DISPLAY_SCREEN_SETTING_KEY,
    SNAP_MIDDLE_EQUAL_ANGLE_ONLY_SETTING_KEY,
    SURFACE_TEXTURE_PROVIDER_GPT_5_6_TERRA,
    SURFACE_TEXTURE_PROVIDER_SETTING_KEY,
    UNUSED_FACE_REMOVAL_SETTING_KEY,
    USE_HALF_MESH_TEXTURE_PREFIX_SETTING_KEY,
    USE_UV_RAYCAST_FOR_OBJECT_GENERATION_SETTING_KEY,
    WALL_VERTEX_UPDATE_DELAY_SECONDS_SETTING_KEY,
    WALL_VERTEX_UPDATE_DELAY_STEP_SECONDS,
    Fullscreen3DViewerScreenOption,
    GenerationServiceSettings,
    SettingsWidget,
    fullscreen_3d_viewer_screen_id,
    read_atlas_display_screen_id,
    read_automatic_atlas_creation,
    read_automatic_atlas_texture_resolution,
    read_automatic_atlas_texture_sort_by_pbr,
    read_canvas_3d_navigation_toggle_hotkey,
    read_first_person_navigation_mode,
    read_generation_display_screen_id,
    read_ignore_top_down_ceiling,
    read_mesh_edit_update_delay_seconds,
    read_minimum_face_visibility_percentage,
    read_scene_3d_display_screen_id,
    read_snap_middle_equal_angle_only,
    read_unused_face_removal,
    read_use_half_mesh_texture_prefix,
    read_use_uv_raycast_for_object_generation,
    read_wall_vertex_update_delay_seconds,
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

    def test_controls_are_organized_into_clear_settings_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            widget = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )

            self.assertIs(
                widget.settings_scroll_area.widget(),
                widget.settings_content,
            )
            expected_sections = (
                (
                    widget.api_credentials_group,
                    "API credentials",
                    (
                        widget.meshy_api_key_edit,
                        widget.meshy_key_status_label,
                        widget.openai_api_key_edit,
                        widget.openai_key_status_label,
                    ),
                ),
                (
                    widget.display_settings_group,
                    "Displays",
                    (
                        widget.scene_3d_display_screen_combo,
                        widget.generation_display_screen_combo,
                        widget.jobs_window_screen_combo,
                        widget.atlas_display_screen_combo,
                    ),
                ),
                (
                    widget.canvas_settings_group,
                    "Canvas",
                    (
                        widget.canvas_3d_navigation_toggle_hotkey_edit,
                        widget.first_person_navigation_combo,
                        widget.ignore_top_down_ceiling_checkbox,
                        widget.snap_middle_equal_angle_only_checkbox,
                        widget.mesh_edit_update_delay_spinbox,
                        widget.wall_vertex_update_delay_spinbox,
                    ),
                ),
                (
                    widget.object_generation_settings_group,
                    "Object generation",
                    (
                        widget.unused_face_removal_checkbox,
                        widget.use_uv_raycast_for_object_generation_checkbox,
                        widget.minimum_face_visibility_percentage_spinbox,
                    ),
                ),
                (
                    widget.atlas_automation_settings_group,
                    "Atlas automation",
                    (
                        widget.automatic_atlas_creation_checkbox,
                        widget.automatic_atlas_texture_sort_by_pbr_checkbox,
                        widget.use_half_mesh_texture_prefix_checkbox,
                        widget.automatic_atlas_texture_resolution_combo,
                    ),
                ),
            )
            for group, title, controls in expected_sections:
                with self.subTest(section=title):
                    self.assertIsInstance(group, QGroupBox)
                    self.assertEqual(group.title(), title)
                    self.assertTrue(widget.settings_content.isAncestorOf(group))
                    for control in controls:
                        self.assertTrue(group.isAncestorOf(control))

            security_note = widget.findChild(QLabel, "api_key_security_note")
            availability_note = widget.findChild(
                QLabel,
                "meshy_availability_note",
            )
            self.assertIsNotNone(security_note)
            self.assertIsNotNone(availability_note)
            assert security_note is not None and availability_note is not None
            self.assertTrue(
                widget.api_credentials_group.isAncestorOf(security_note)
            )
            self.assertTrue(
                widget.api_credentials_group.isAncestorOf(availability_note)
            )
            canvas_form = widget.canvas_settings_group.layout()
            self.assertIsInstance(canvas_form, QFormLayout)
            assert isinstance(canvas_form, QFormLayout)
            self.assertEqual(
                canvas_form.labelForField(
                    widget.first_person_navigation_combo
                ).text(),
                "First person navigation",
            )
            self.assertEqual(
                canvas_form.labelForField(
                    widget.ignore_top_down_ceiling_checkbox
                ).text(),
                "Ignore top-down ceiling",
            )
            widget.dispose()

    def test_first_person_navigation_mode_persists_and_emits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            combo = widget.first_person_navigation_combo
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )

            self.assertEqual(
                FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY,
                "navigation/first_person_navigation_mode",
            )
            self.assertEqual(
                DEFAULT_FIRST_PERSON_NAVIGATION_MODE,
                FIRST_PERSON_NAVIGATION_MODE_GRAVITY,
            )
            self.assertEqual(
                tuple(
                    (combo.itemText(index), combo.itemData(index))
                    for index in range(combo.count())
                ),
                FIRST_PERSON_NAVIGATION_MODE_OPTIONS,
            )
            self.assertEqual(
                widget.get_settings().first_person_navigation_mode,
                FIRST_PERSON_NAVIGATION_MODE_GRAVITY,
            )

            combo.setCurrentIndex(
                combo.findData(FIRST_PERSON_NAVIGATION_MODE_NOCLIP)
            )

            self.assertEqual(
                application_settings.get(
                    FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY
                ),
                FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
            )
            self.assertEqual(
                widget.get_settings().first_person_navigation_mode,
                FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
            )
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertEqual(
                restored.get_settings().first_person_navigation_mode,
                FIRST_PERSON_NAVIGATION_MODE_NOCLIP,
            )
            widget.dispose()
            restored.dispose()

    def test_ignore_top_down_ceiling_defaults_persists_and_emits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            checkbox = widget.ignore_top_down_ceiling_checkbox
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )

            self.assertTrue(DEFAULT_IGNORE_TOP_DOWN_CEILING)
            self.assertTrue(checkbox.isChecked())
            self.assertTrue(widget.get_settings().ignore_top_down_ceiling)

            checkbox.setChecked(False)

            self.assertFalse(
                application_settings.get(
                    IGNORE_TOP_DOWN_CEILING_SETTING_KEY
                )
            )
            self.assertFalse(widget.get_settings().ignore_top_down_ceiling)
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertFalse(
                restored.get_settings().ignore_top_down_ceiling
            )
            widget.dispose()
            restored.dispose()

    def test_ignore_top_down_ceiling_rejects_malformed_values(self) -> None:
        for value in (0, 1, "true", None):
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Top-down ceiling suppression",
                ):
                    GenerationServiceSettings(
                        ignore_top_down_ceiling=value  # type: ignore[arg-type]
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in (0, 1, "true", None):
                with self.subTest(persisted_value=value):
                    application_settings.set(
                        IGNORE_TOP_DOWN_CEILING_SETTING_KEY,
                        value,
                    )
                    self.assertTrue(
                        read_ignore_top_down_ceiling(application_settings)
                    )

    def test_first_person_navigation_mode_rejects_malformed_values(
        self,
    ) -> None:
        self.assertEqual(
            GenerationServiceSettings(
                first_person_navigation_mode=" Gravity "
            ).first_person_navigation_mode,
            FIRST_PERSON_NAVIGATION_MODE_GRAVITY,
        )

        for value in ("", "fly", 1, None, []):
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "First-person navigation mode",
                ):
                    GenerationServiceSettings(
                        first_person_navigation_mode=(
                            value  # type: ignore[arg-type]
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            application_settings.set(
                FIRST_PERSON_NAVIGATION_MODE_SETTING_KEY,
                "fly",
            )

            self.assertEqual(
                read_first_person_navigation_mode(application_settings),
                FIRST_PERSON_NAVIGATION_MODE_GRAVITY,
            )

    def test_snap_middle_equal_angle_setting_persists_and_emits(self) -> None:
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

            self.assertTrue(DEFAULT_SNAP_MIDDLE_EQUAL_ANGLE_ONLY)
            self.assertTrue(
                widget.snap_middle_equal_angle_only_checkbox.isChecked()
            )
            self.assertTrue(
                widget.get_settings().snap_middle_equal_angle_only
            )

            widget.snap_middle_equal_angle_only_checkbox.setChecked(False)

            self.assertFalse(
                application_settings.get(
                    SNAP_MIDDLE_EQUAL_ANGLE_ONLY_SETTING_KEY
                )
            )
            self.assertFalse(
                widget.get_settings().snap_middle_equal_angle_only
            )
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertFalse(
                restored.get_settings().snap_middle_equal_angle_only
            )
            widget.dispose()
            restored.dispose()

    def test_snap_middle_equal_angle_setting_rejects_malformed_values(
        self,
    ) -> None:
        for value in (1, "true", None, []):
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Snap-to-middle equal-angle",
                ):
                    GenerationServiceSettings(
                        snap_middle_equal_angle_only=(
                            value  # type: ignore[arg-type]
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            application_settings.set(
                SNAP_MIDDLE_EQUAL_ANGLE_ONLY_SETTING_KEY,
                "yes",
            )

            self.assertTrue(
                read_snap_middle_equal_angle_only(application_settings)
            )

    def test_atlas_display_model_defaults_normalizes_and_validates(self) -> None:
        self.assertIsNone(GenerationServiceSettings().atlas_display_screen_id)
        self.assertEqual(
            GenerationServiceSettings(
                atlas_display_screen_id="  screen:atlas  "
            ).atlas_display_screen_id,
            "screen:atlas",
        )

        for value in (False, 1, []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Atlas display"):
                    GenerationServiceSettings(
                        atlas_display_screen_id=(
                            value  # type: ignore[arg-type]
                        )
                    )

    def test_scene_and_generation_display_models_normalize_and_validate(
        self,
    ) -> None:
        defaults = GenerationServiceSettings()
        self.assertIsNone(defaults.scene_3d_display_screen_id)
        self.assertIsNone(defaults.generation_display_screen_id)
        settings = GenerationServiceSettings(
            scene_3d_display_screen_id="  screen:scene  ",
            generation_display_screen_id="  screen:generation  ",
        )
        self.assertEqual(
            settings.scene_3d_display_screen_id,
            "screen:scene",
        )
        self.assertEqual(
            settings.generation_display_screen_id,
            "screen:generation",
        )

        for field_name, error_text in (
            ("scene_3d_display_screen_id", "3D scene display"),
            ("generation_display_screen_id", "Generation display"),
        ):
            for value in (False, 1, []):
                with self.subTest(field=field_name, value=value):
                    with self.assertRaisesRegex(ValueError, error_text):
                        GenerationServiceSettings(
                            **{field_name: value}  # type: ignore[arg-type]
                        )

    def test_automatic_atlas_creation_defaults_persists_and_restores(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            checkbox = widget.automatic_atlas_creation_checkbox
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )

            self.assertFalse(DEFAULT_AUTOMATIC_ATLAS_CREATION)
            self.assertFalse(checkbox.isChecked())
            self.assertFalse(widget.get_settings().automatic_atlas_creation)

            checkbox.setChecked(True)

            self.assertTrue(
                application_settings.get(AUTOMATIC_ATLAS_CREATION_SETTING_KEY)
            )
            self.assertTrue(widget.get_settings().automatic_atlas_creation)
            self.assertEqual(emitted_changes, [True])
            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertTrue(restored.get_settings().automatic_atlas_creation)

    def test_automatic_atlas_creation_rejects_malformed_values(self) -> None:
        for value in (0, 1, "true", None):
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Automatic atlas creation",
                ):
                    GenerationServiceSettings(
                        automatic_atlas_creation=value  # type: ignore[arg-type]
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in (0, 1, "true", None):
                with self.subTest(persisted_value=value):
                    application_settings.set(
                        AUTOMATIC_ATLAS_CREATION_SETTING_KEY,
                        value,
                    )
                    self.assertFalse(
                        read_automatic_atlas_creation(application_settings)
                    )

    def test_automatic_atlas_resolution_defaults_persists_and_emits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            combo = widget.automatic_atlas_texture_resolution_combo
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )

            self.assertEqual(
                DEFAULT_AUTOMATIC_ATLAS_TEXTURE_RESOLUTION,
                512,
            )
            self.assertEqual(
                widget.get_settings().automatic_atlas_texture_resolution,
                DEFAULT_AUTOMATIC_ATLAS_TEXTURE_RESOLUTION,
            )
            self.assertEqual(
                tuple(combo.itemData(index) for index in range(combo.count())),
                AUTOMATIC_ATLAS_TEXTURE_RESOLUTIONS,
            )

            combo.setCurrentIndex(combo.findData(1024))

            self.assertEqual(
                application_settings.get(
                    AUTOMATIC_ATLAS_TEXTURE_RESOLUTION_SETTING_KEY
                ),
                1024,
            )
            self.assertEqual(
                widget.get_settings().automatic_atlas_texture_resolution,
                1024,
            )
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertEqual(
                restored.get_settings().automatic_atlas_texture_resolution,
                1024,
            )

    def test_automatic_atlas_resolution_rejects_malformed_values(self) -> None:
        invalid_values: tuple[object, ...] = (
            True,
            "1024",
            1024.0,
            256,
            2048,
            None,
        )
        for value in invalid_values:
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Automatic Atlas texture resolution",
                ):
                    GenerationServiceSettings(
                        automatic_atlas_texture_resolution=(
                            value  # type: ignore[arg-type]
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in invalid_values:
                with self.subTest(persisted_value=value):
                    application_settings.set(
                        AUTOMATIC_ATLAS_TEXTURE_RESOLUTION_SETTING_KEY,
                        value,
                    )
                    self.assertEqual(
                        read_automatic_atlas_texture_resolution(
                            application_settings
                        ),
                        DEFAULT_AUTOMATIC_ATLAS_TEXTURE_RESOLUTION,
                    )

    def test_automatic_atlas_resolution_accepts_supported_values(self) -> None:
        self.assertEqual(
            AUTOMATIC_ATLAS_TEXTURE_RESOLUTIONS,
            (512, 1024),
        )
        for resolution in AUTOMATIC_ATLAS_TEXTURE_RESOLUTIONS:
            with self.subTest(resolution=resolution):
                self.assertEqual(
                    GenerationServiceSettings(
                        automatic_atlas_texture_resolution=resolution
                    ).automatic_atlas_texture_resolution,
                    resolution,
                )

    def test_automatic_atlas_pbr_sort_defaults_persists_and_restores(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            checkbox = widget.automatic_atlas_texture_sort_by_pbr_checkbox
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )

            self.assertFalse(DEFAULT_AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR)
            self.assertFalse(checkbox.isChecked())
            self.assertFalse(
                widget.get_settings().automatic_atlas_texture_sort_by_pbr
            )

            checkbox.setChecked(True)

            self.assertTrue(
                application_settings.get(
                    AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR_SETTING_KEY
                )
            )
            self.assertTrue(
                widget.get_settings().automatic_atlas_texture_sort_by_pbr
            )
            self.assertEqual(emitted_changes, [True])
            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertTrue(
                restored.get_settings().automatic_atlas_texture_sort_by_pbr
            )

    def test_automatic_atlas_pbr_sort_rejects_malformed_values(self) -> None:
        for value in (0, 1, "true", None):
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Automatic Atlas PBR sorting",
                ):
                    GenerationServiceSettings(
                        automatic_atlas_texture_sort_by_pbr=(
                            value  # type: ignore[arg-type]
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in (0, 1, "true", None):
                with self.subTest(persisted_value=value):
                    application_settings.set(
                        AUTOMATIC_ATLAS_TEXTURE_SORT_BY_PBR_SETTING_KEY,
                        value,
                    )
                    self.assertFalse(
                        read_automatic_atlas_texture_sort_by_pbr(
                            application_settings
                        )
                    )

    def test_half_mesh_texture_prefix_defaults_persists_and_restores(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            checkbox = widget.use_half_mesh_texture_prefix_checkbox
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )

            self.assertFalse(DEFAULT_USE_HALF_MESH_TEXTURE_PREFIX)
            self.assertFalse(checkbox.isChecked())
            self.assertFalse(
                widget.get_settings().use_half_mesh_texture_prefix
            )
            self.assertIn(
                "Use [HALF] half-mesh texture prefix",
                [label.text() for label in widget.findChildren(QLabel)],
            )

            checkbox.setChecked(True)

            self.assertTrue(
                application_settings.get(
                    USE_HALF_MESH_TEXTURE_PREFIX_SETTING_KEY
                )
            )
            self.assertTrue(
                widget.get_settings().use_half_mesh_texture_prefix
            )
            self.assertEqual(emitted_changes, [True])
            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertTrue(
                restored.get_settings().use_half_mesh_texture_prefix
            )

    def test_half_mesh_texture_prefix_rejects_malformed_values(self) -> None:
        for value in (0, 1, "true", None):
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Half-mesh texture prefix",
                ):
                    GenerationServiceSettings(
                        use_half_mesh_texture_prefix=(
                            value  # type: ignore[arg-type]
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in (0, 1, "true", None):
                with self.subTest(persisted_value=value):
                    application_settings.set(
                        USE_HALF_MESH_TEXTURE_PREFIX_SETTING_KEY,
                        value,
                    )
                    self.assertFalse(
                        read_use_half_mesh_texture_prefix(application_settings)
                    )

    def test_api_key_fields_and_workspace_display_selectors_are_visible(
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
                hasattr(widget, "scene_3d_display_screen_combo")
            )
            self.assertTrue(
                hasattr(widget, "generation_display_screen_combo")
            )
            self.assertTrue(hasattr(widget, "atlas_display_screen_combo"))
            self.assertFalse(
                hasattr(widget, "fullscreen_3d_viewer_screen_combo")
            )
            self.assertTrue(
                hasattr(
                    widget,
                    "canvas_3d_navigation_toggle_hotkey_edit",
                )
            )
            self.assertTrue(
                hasattr(widget, "mesh_edit_update_delay_spinbox")
            )
            self.assertTrue(
                hasattr(widget, "wall_vertex_update_delay_spinbox")
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

    def test_workspace_display_ids_can_be_read_without_settings_file_io(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            for combo in (
                widget.scene_3d_display_screen_combo,
                widget.generation_display_screen_combo,
            ):
                combo.addItem("Cached display", "screen:cached")
                combo.setCurrentIndex(combo.findData("screen:cached"))

            with patch.object(
                application_settings,
                "get",
                side_effect=AssertionError("unexpected settings-file read"),
            ):
                scene_screen_id = widget.get_scene_3d_display_screen_id()
                generation_screen_id = (
                    widget.get_generation_display_screen_id()
                )

            self.assertEqual(scene_screen_id, "screen:cached")
            self.assertEqual(generation_screen_id, "screen:cached")

    def test_atlas_display_persists_restores_and_is_cached(self) -> None:
        options = (
            Fullscreen3DViewerScreenOption(
                screen_id="monitor:Acme|Atlas|002",
                label="Atlas display (1920 x 1080)",
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
                combo = widget.atlas_display_screen_combo
                emitted_changes: list[bool] = []
                widget.settings_changed.connect(
                    lambda: emitted_changes.append(True)
                )

                self.assertEqual(combo.itemText(0), "None")
                self.assertIsNone(combo.itemData(0))
                self.assertIsNone(
                    widget.get_settings().atlas_display_screen_id
                )

                combo.setCurrentIndex(combo.findData(options[0].screen_id))

                self.assertEqual(
                    application_settings.get(ATLAS_DISPLAY_SCREEN_SETTING_KEY),
                    options[0].screen_id,
                )
                self.assertEqual(
                    read_atlas_display_screen_id(application_settings),
                    options[0].screen_id,
                )
                self.assertEqual(
                    widget.get_settings().atlas_display_screen_id,
                    options[0].screen_id,
                )
                self.assertEqual(emitted_changes, [True])

                with patch.object(
                    application_settings,
                    "get",
                    side_effect=AssertionError("unexpected settings-file read"),
                ):
                    selected_id = widget.get_atlas_display_screen_id()
                self.assertEqual(selected_id, options[0].screen_id)

                restored = SettingsWidget(
                    application_settings=_build_test_settings(
                        temporary_directory
                    ),
                    environment={},
                )
                self.assertEqual(
                    restored.atlas_display_screen_combo.currentData(),
                    options[0].screen_id,
                )

    def test_atlas_display_reader_ignores_malformed_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in (False, 1, []):
                with self.subTest(value=value):
                    application_settings.set(
                        ATLAS_DISPLAY_SCREEN_SETTING_KEY,
                        value,
                    )
                    self.assertIsNone(
                        read_atlas_display_screen_id(application_settings)
                    )

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

        self.assertEqual(
            settings.scene_3d_display_screen_id,
            "viewer-screen",
        )
        self.assertEqual(settings.jobs_window_screen_id, "jobs-screen")
        self.assertEqual(settings.mesh_edit_update_delay_seconds, 1.25)
        self.assertEqual(
            settings.wall_vertex_update_delay_seconds,
            DEFAULT_WALL_VERTEX_UPDATE_DELAY_SECONDS,
        )
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

    def test_mesh_edit_update_delay_persists_and_emits_changes(self) -> None:
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
                MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY,
                "canvas/doorway_mesh_update_delay_seconds",
            )
            self.assertEqual(
                widget.get_settings().mesh_edit_update_delay_seconds,
                DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS,
            )
            self.assertFalse(
                widget.mesh_edit_update_delay_spinbox.keyboardTracking()
            )
            self.assertEqual(
                widget.mesh_edit_update_delay_spinbox.minimum(),
                MIN_MESH_EDIT_UPDATE_DELAY_SECONDS,
            )
            self.assertEqual(
                widget.mesh_edit_update_delay_spinbox.maximum(),
                MAX_MESH_EDIT_UPDATE_DELAY_SECONDS,
            )
            self.assertEqual(
                widget.mesh_edit_update_delay_spinbox.singleStep(),
                MESH_EDIT_UPDATE_DELAY_STEP_SECONDS,
            )
            self.assertEqual(
                widget.mesh_edit_update_delay_spinbox.suffix(),
                " s",
            )
            self.assertIn(
                "doorway or window edit",
                widget.mesh_edit_update_delay_spinbox.toolTip(),
            )
            self.assertIn(
                "wall dimension edit",
                widget.mesh_edit_update_delay_spinbox.toolTip(),
            )
            form_layout = widget.canvas_settings_group.layout()
            self.assertIsInstance(form_layout, QFormLayout)
            assert isinstance(form_layout, QFormLayout)
            self.assertEqual(
                form_layout.labelForField(
                    widget.mesh_edit_update_delay_spinbox
                ).text(),
                "Mesh edit update delay",
            )
            widget.mesh_edit_update_delay_spinbox.setValue(2.4)

            self.assertEqual(
                application_settings.get(
                    MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY
                ),
                2.4,
            )
            self.assertEqual(
                widget.get_settings().mesh_edit_update_delay_seconds,
                2.4,
            )
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertEqual(
                restored.get_settings().mesh_edit_update_delay_seconds,
                2.4,
            )

    def test_mesh_edit_update_delay_rejects_malformed_values(self) -> None:
        invalid_values: tuple[object, ...] = (
            True,
            "1.0",
            float("nan"),
            float("inf"),
            float("-inf"),
            MIN_MESH_EDIT_UPDATE_DELAY_SECONDS - 0.01,
            MAX_MESH_EDIT_UPDATE_DELAY_SECONDS + 0.01,
        )
        for value in invalid_values:
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Mesh edit update delay",
                ):
                    GenerationServiceSettings(
                        mesh_edit_update_delay_seconds=(
                            value  # type: ignore[arg-type]
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in invalid_values:
                with self.subTest(persisted_value=value):
                    application_settings.set(
                        MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY,
                        value,
                    )
                    self.assertEqual(
                        read_mesh_edit_update_delay_seconds(
                            application_settings
                        ),
                        DEFAULT_MESH_EDIT_UPDATE_DELAY_SECONDS,
                    )

    def test_mesh_edit_update_delay_accepts_range_boundaries(self) -> None:
        for value in (
            MIN_MESH_EDIT_UPDATE_DELAY_SECONDS,
            MAX_MESH_EDIT_UPDATE_DELAY_SECONDS,
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    GenerationServiceSettings(
                        mesh_edit_update_delay_seconds=value
                    ).mesh_edit_update_delay_seconds,
                    value,
                )
        self.assertIsInstance(
            GenerationServiceSettings(
                mesh_edit_update_delay_seconds=1
            ).mesh_edit_update_delay_seconds,
            float,
        )

    def test_wall_vertex_update_delay_persists_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            application_settings.set(
                MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY,
                2.4,
            )
            widget = SettingsWidget(
                application_settings=application_settings,
                environment={},
            )
            emitted_changes: list[bool] = []
            widget.settings_changed.connect(
                lambda: emitted_changes.append(True)
            )

            self.assertEqual(
                WALL_VERTEX_UPDATE_DELAY_SECONDS_SETTING_KEY,
                "canvas/wall_vertex_update_delay_seconds",
            )
            self.assertEqual(
                widget.get_settings().wall_vertex_update_delay_seconds,
                DEFAULT_WALL_VERTEX_UPDATE_DELAY_SECONDS,
            )
            self.assertEqual(
                widget.get_settings().mesh_edit_update_delay_seconds,
                2.4,
            )
            self.assertFalse(
                widget.wall_vertex_update_delay_spinbox.keyboardTracking()
            )
            self.assertEqual(
                widget.wall_vertex_update_delay_spinbox.minimum(),
                MIN_WALL_VERTEX_UPDATE_DELAY_SECONDS,
            )
            self.assertEqual(
                widget.wall_vertex_update_delay_spinbox.maximum(),
                MAX_WALL_VERTEX_UPDATE_DELAY_SECONDS,
            )
            self.assertEqual(
                widget.wall_vertex_update_delay_spinbox.singleStep(),
                WALL_VERTEX_UPDATE_DELAY_STEP_SECONDS,
            )
            self.assertEqual(
                widget.wall_vertex_update_delay_spinbox.suffix(),
                " s",
            )
            self.assertIn(
                "adding a wall vertex",
                widget.wall_vertex_update_delay_spinbox.toolTip(),
            )
            form_layout = widget.canvas_settings_group.layout()
            self.assertIsInstance(form_layout, QFormLayout)
            assert isinstance(form_layout, QFormLayout)
            self.assertEqual(
                form_layout.labelForField(
                    widget.wall_vertex_update_delay_spinbox
                ).text(),
                "Wall vertex update delay",
            )

            widget.wall_vertex_update_delay_spinbox.setValue(18.5)

            self.assertEqual(
                application_settings.get(
                    WALL_VERTEX_UPDATE_DELAY_SECONDS_SETTING_KEY
                ),
                18.5,
            )
            self.assertEqual(
                application_settings.get(
                    MESH_EDIT_UPDATE_DELAY_SECONDS_SETTING_KEY
                ),
                2.4,
            )
            self.assertEqual(
                widget.get_settings().wall_vertex_update_delay_seconds,
                18.5,
            )
            self.assertEqual(emitted_changes, [True])

            restored = SettingsWidget(
                application_settings=_build_test_settings(temporary_directory),
                environment={},
            )
            self.assertEqual(
                restored.get_settings().wall_vertex_update_delay_seconds,
                18.5,
            )

    def test_wall_vertex_update_delay_rejects_malformed_values(self) -> None:
        invalid_values: tuple[object, ...] = (
            True,
            "15.0",
            float("nan"),
            float("inf"),
            float("-inf"),
            MIN_WALL_VERTEX_UPDATE_DELAY_SECONDS - 0.01,
            MAX_WALL_VERTEX_UPDATE_DELAY_SECONDS + 0.01,
        )
        for value in invalid_values:
            with self.subTest(model_value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "Wall vertex update delay",
                ):
                    GenerationServiceSettings(
                        wall_vertex_update_delay_seconds=(
                            value  # type: ignore[arg-type]
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for value in invalid_values:
                with self.subTest(persisted_value=value):
                    application_settings.set(
                        WALL_VERTEX_UPDATE_DELAY_SECONDS_SETTING_KEY,
                        value,
                    )
                    self.assertEqual(
                        read_wall_vertex_update_delay_seconds(
                            application_settings
                        ),
                        DEFAULT_WALL_VERTEX_UPDATE_DELAY_SECONDS,
                    )

    def test_wall_vertex_update_delay_accepts_range_boundaries(self) -> None:
        for value in (
            MIN_WALL_VERTEX_UPDATE_DELAY_SECONDS,
            MAX_WALL_VERTEX_UPDATE_DELAY_SECONDS,
        ):
            with self.subTest(value=value):
                self.assertEqual(
                    GenerationServiceSettings(
                        wall_vertex_update_delay_seconds=value
                    ).wall_vertex_update_delay_seconds,
                    value,
                )
        self.assertIsInstance(
            GenerationServiceSettings(
                wall_vertex_update_delay_seconds=15
            ).wall_vertex_update_delay_seconds,
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
            "Ctrl+Z",
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

    def test_workspace_displays_persist_selected_screen_ids(self) -> None:
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
                scene_combo = widget.scene_3d_display_screen_combo
                generation_combo = widget.generation_display_screen_combo
                display_form = widget.display_settings_group.layout()
                assert isinstance(display_form, QFormLayout)

                self.assertEqual(
                    display_form.labelForField(scene_combo).text(),
                    "3D scene display",
                )
                self.assertEqual(
                    display_form.labelForField(generation_combo).text(),
                    "Generation display",
                )
                for combo in (scene_combo, generation_combo):
                    self.assertEqual(combo.itemText(0), "None")
                    self.assertIsNone(combo.itemData(0))
                    self.assertEqual(
                        [
                            combo.itemData(index)
                            for index in range(combo.count())
                        ],
                        [None, options[0].screen_id, options[1].screen_id],
                    )

                scene_combo.setCurrentIndex(1)
                generation_combo.setCurrentIndex(2)

                self.assertEqual(
                    application_settings.get(
                        SCENE_3D_DISPLAY_SCREEN_SETTING_KEY
                    ),
                    options[0].screen_id,
                )
                self.assertEqual(
                    application_settings.get(
                        GENERATION_DISPLAY_SCREEN_SETTING_KEY
                    ),
                    options[1].screen_id,
                )
                self.assertEqual(
                    widget.get_settings().scene_3d_display_screen_id,
                    options[0].screen_id,
                )
                self.assertEqual(
                    widget.get_settings().generation_display_screen_id,
                    options[1].screen_id,
                )

                restored = SettingsWidget(
                    application_settings=_build_test_settings(
                        temporary_directory
                    ),
                    environment={},
                )
                self.assertEqual(
                    restored.scene_3d_display_screen_combo.currentData(),
                    options[0].screen_id,
                )
                self.assertEqual(
                    restored.generation_display_screen_combo.currentData(),
                    options[1].screen_id,
                )

    def test_scene_display_uses_legacy_value_only_when_new_key_is_absent(
        self,
    ) -> None:
        options = (
            Fullscreen3DViewerScreenOption(
                screen_id="screen:legacy",
                label="Legacy display",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            application_settings.set(
                FULLSCREEN_3D_VIEWER_SCREEN_SETTING_KEY,
                options[0].screen_id,
            )
            with patch(
                "housemaker.settings_widget."
                "connected_fullscreen_3d_viewer_display_options",
                return_value=options,
            ):
                migrated = SettingsWidget(
                    application_settings=application_settings,
                    environment={},
                )
                self.assertEqual(
                    migrated.scene_3d_display_screen_combo.currentData(),
                    options[0].screen_id,
                )
                self.assertEqual(
                    read_scene_3d_display_screen_id(application_settings),
                    options[0].screen_id,
                )

                application_settings.set(
                    SCENE_3D_DISPLAY_SCREEN_SETTING_KEY,
                    None,
                )
                explicitly_embedded = SettingsWidget(
                    application_settings=application_settings,
                    environment={},
                )

            self.assertIsNone(
                explicitly_embedded.scene_3d_display_screen_combo.currentData()
            )
            self.assertIsNone(
                read_scene_3d_display_screen_id(application_settings)
            )

    def test_workspace_display_readers_ignore_malformed_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application_settings = _build_test_settings(temporary_directory)
            for setting_key, reader in (
                (
                    SCENE_3D_DISPLAY_SCREEN_SETTING_KEY,
                    read_scene_3d_display_screen_id,
                ),
                (
                    GENERATION_DISPLAY_SCREEN_SETTING_KEY,
                    read_generation_display_screen_id,
                ),
            ):
                for value in (False, 1, []):
                    with self.subTest(key=setting_key, value=value):
                        application_settings.set(setting_key, value)
                        self.assertIsNone(reader(application_settings))

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

    def test_missing_saved_scene_display_safely_uses_none(self) -> None:
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
                widget.scene_3d_display_screen_combo.currentData()
            )
            self.assertIsNone(
                widget.get_settings().scene_3d_display_screen_id
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
