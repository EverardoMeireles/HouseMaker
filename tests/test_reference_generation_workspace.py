# ### Environment setup ###
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import ClassVar
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QComboBox

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.generation_workspace import (
    GenerationWorkspace,
    _ArchitecturalTrimEditingTarget,
)
from housemaker.merged_generation_workspace import (
    REFERENCE_GENERATION_NEGATIVE_PROMPT_SETTINGS_KEY,
    REFERENCE_GENERATION_PROMPT_SETTINGS_KEY,
    REFERENCE_GENERATION_STRENGTH_SETTINGS_KEY,
    REFERENCE_GENERATION_VIEW_SETTINGS_KEY,
    MergedGenerationWorkspace,
    _ReferenceGenerationThread,
)
from housemaker.reference_generation_models import (
    DEFAULT_REFERENCE_NEGATIVE_PROMPT,
    REFERENCE_GENERATION_MODEL_OPTIONS,
    REFERENCE_GENERATION_MODEL_SETTINGS_KEY,
    REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
)
from housemaker.reference_view_presets import (
    DEFAULT_REFERENCE_VIEW_PRESETS,
    ReferenceViewPreset,
    read_reference_view_presets,
    save_reference_view_presets,
)
from housemaker.settings_widget import SettingsWidget
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace

# ### Fixtures ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


class _ControlledReferenceThread(QObject):
    """Complete provider work explicitly without a network, model or real thread."""

    progress_changed = Signal(str, int)
    finished = Signal()
    created: ClassVar[list[_ControlledReferenceThread]] = []

    def __init__(
        self,
        source_bgra,
        prompt,
        model,
        *,
        api_key,
        strength,
        negative_prompt=DEFAULT_REFERENCE_NEGATIVE_PROMPT,
        parent=None,
    ):
        super().__init__(parent)
        self.source_bgra = source_bgra.copy()
        self.prompt = prompt
        self.model = model
        self.api_key = api_key
        self.strength = strength
        self.negative_prompt = negative_prompt
        self.result = None
        self.error_message = None
        self.was_cancelled = False
        self.interrupted = False
        self.created.append(self)

    def start(self, *_args):
        pass

    def requestInterruption(self):
        self.interrupted = True

    def isRunning(self):
        return False

    def wait(self, *_args):
        return True

    def complete(self, image=None, error=None):
        self.result = image
        self.error_message = error
        self.finished.emit()


def _opaque_reference(value: int = 70) -> np.ndarray:
    image = np.full((32, 48, 4), value, dtype=np.uint8)
    image[:, :, 3] = 255
    return image


def _delete_test_widgets(*widgets) -> None:
    """Destroy Qt ownership trees before their Python wrappers are collected."""

    for widget in widgets:
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    _qt_application.processEvents()


# ### Reference generation workspace tests ###
class ReferenceGenerationWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.settings = ApplicationSettingsStore(self.root / "settings.json")
        self.job_manager = GenerationJobManager()
        self.objects = GenerationWorkspace(
            asset_directory=self.root / "objects", job_manager=self.job_manager
        )
        self.surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=self.root / "surfaces",
            application_settings=self.settings,
            job_manager=self.job_manager,
            shared_controls=self.objects.get_shared_controls(),
        )
        self.workspace = MergedGenerationWorkspace(
            self.surfaces,
            self.objects,
            self.job_manager,
            application_settings=self.settings,
        )
        self.settings_widget = SettingsWidget(
            application_settings=self.settings, environment={}
        )
        self.settings_widget.settings_changed.connect(
            self._apply_reference_generation_model
        )
        _ControlledReferenceThread.created = []
        self.thread_patch = patch(
            "housemaker.merged_generation_workspace._ReferenceGenerationThread",
            _ControlledReferenceThread,
        )
        self.thread_patch.start()

    def tearDown(self):
        self.settings_widget.settings_changed.disconnect(
            self._apply_reference_generation_model
        )
        self.settings_widget.dispose()
        self.workspace.shutdown()
        self.objects.shutdown()
        self.surfaces.shutdown()
        _ControlledReferenceThread.created.clear()
        _delete_test_widgets(self.settings_widget, self.workspace)
        self.thread_patch.stop()
        self.temporary_directory.cleanup()

    def _apply_reference_generation_model(self):
        self.workspace.set_reference_generation_model(
            self.settings_widget.get_settings().reference_generation_model
        )

    def _load_video(self):
        video_path = self.root / "reference.avi"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
        )
        self.assertTrue(writer.isOpened())
        for value in (40, 80, 120):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        writer.release()
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))
        self.workspace.video_view.set_strokes(
            [MaskStroke(MASK_MODE_PAINT, 0.2, (MaskPoint(0.5, 0.5),))]
        )
        self.workspace.sync_shared_controls()

    def _start(self, source=None):
        if source is not None:
            self.workspace._activate_temporary_object_reference(
                source, source_label="Loaded image"
            )
        self.workspace.reference_generation_view_combo.setCurrentIndex(
            self.workspace.reference_generation_view_combo.findData("")
        )
        self.workspace.reference_generation_prompt_edit.setText(
            "A plain wooden chair without the fabric cover"
        )
        self.workspace.generate_reference_button.click()
        self.assertIsNotNone(self.workspace._reference_generation_runtime)
        return _ControlledReferenceThread.created[-1]

    @contextmanager
    def _reopened_workspace(self):
        reopened_settings = ApplicationSettingsStore(self.root / "settings.json")
        objects = GenerationWorkspace(asset_directory=self.root / "reopened_objects")
        surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=self.root / "reopened_surfaces",
            application_settings=reopened_settings,
            shared_controls=objects.get_shared_controls(),
        )
        workspace = MergedGenerationWorkspace(
            surfaces,
            objects,
            GenerationJobManager(),
            application_settings=reopened_settings,
        )
        try:
            yield workspace
        finally:
            workspace.shutdown()
            objects.shutdown()
            surfaces.shutdown()
            _delete_test_widgets(workspace)

    def test_settings_model_selection_updates_sdxl_controls_without_generation_selector(
        self,
    ):
        combo = self.settings_widget.reference_generation_model_combo
        self.assertEqual(
            tuple((combo.itemText(i), combo.itemData(i)) for i in range(combo.count())),
            REFERENCE_GENERATION_MODEL_OPTIONS,
        )
        self.assertFalse(hasattr(self.workspace, "reference_generation_model_combo"))
        self.assertIsNone(
            self.workspace.findChild(QComboBox, "reference_generation_model_combo")
        )
        self.assertTrue(self.workspace.reference_generation_strength_control.isHidden())
        self.assertTrue(
            self.workspace.reference_generation_negative_prompt_control.isHidden()
        )
        combo.setCurrentIndex(
            combo.findData(REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION)
        )
        self.assertFalse(
            self.workspace.reference_generation_strength_control.isHidden()
        )
        self.assertFalse(
            self.workspace.reference_generation_negative_prompt_control.isHidden()
        )
        self.assertEqual(
            self.workspace.reference_generation_negative_prompt_edit.text(),
            DEFAULT_REFERENCE_NEGATIVE_PROMPT.removeprefix(
                "side view, three-quarter view, "
            ),
        )
        self.assertAlmostEqual(
            self.workspace.reference_generation_strength_spinbox.value(), 0.45
        )

        thread = self._start(_opaque_reference())
        self.assertEqual(thread.model, REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION)
        thread.complete(_opaque_reference(190))
        combo.setCurrentIndex(0)
        self.assertTrue(self.workspace.reference_generation_strength_control.isHidden())
        self.assertTrue(
            self.workspace.reference_generation_negative_prompt_control.isHidden()
        )

    def test_view_dropdown_contains_four_defaults_and_write_prompt(self):
        combo = self.workspace.reference_generation_view_combo
        items = tuple(
            (combo.itemText(index), combo.itemData(index))
            for index in range(combo.count())
        )
        self.assertEqual(len(DEFAULT_REFERENCE_VIEW_PRESETS), 4)
        for preset in DEFAULT_REFERENCE_VIEW_PRESETS:
            self.assertIn((preset.name, preset.preset_id), items)
        self.assertIn(("(Write prompt)", ""), items)
        self.assertEqual(combo.count(), 5)
        self.assertEqual(
            items[:4],
            tuple(
                (preset.name, preset.preset_id)
                for preset in DEFAULT_REFERENCE_VIEW_PRESETS
            ),
        )
        self.assertEqual(combo.currentData(), "elevated-three-quarter")
        self.assertFalse(self.workspace.reference_generation_prompt_control.isHidden())
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(),
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
        )

    def test_settings_model_change_preserves_running_request_and_updates_the_next(self):
        combo = self.settings_widget.reference_generation_model_combo
        original_model = combo.currentData()
        thread = self._start(_opaque_reference())

        combo.setCurrentIndex(
            combo.findData(REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION)
        )

        self.assertEqual(thread.model, original_model)
        self.assertIsNotNone(self.workspace._reference_generation_runtime)
        self.assertFalse(self.workspace.reference_generation_strength_spinbox.isEnabled())
        thread.complete(_opaque_reference(190))
        self.assertTrue(self.workspace.reference_generation_strength_spinbox.isEnabled())
        second = self._start()
        self.assertEqual(second.model, REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION)

    def test_prompt_is_always_visible_and_write_prompt_retains_current_text(self):
        combo = self.workspace.reference_generation_view_combo
        self.workspace.reference_generation_prompt_edit.setText("My custom reference")
        self.assertEqual(combo.currentData(), "elevated-three-quarter")
        self.assertFalse(self.workspace.reference_generation_prompt_control.isHidden())
        combo.setCurrentIndex(combo.findData(""))
        self.assertFalse(self.workspace.reference_generation_prompt_control.isHidden())
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(), "My custom reference"
        )
        combo.setCurrentIndex(combo.findData("front"))
        self.assertFalse(self.workspace.reference_generation_prompt_control.isHidden())
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(),
            DEFAULT_REFERENCE_VIEW_PRESETS[1].prompt,
        )

    def test_selected_preset_generates_without_a_typed_prompt(self):
        self.workspace._activate_temporary_object_reference(
            _opaque_reference(), source_label="Loaded image"
        )
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        self.workspace.generate_reference_button.click()
        self.assertIsNotNone(self.workspace._reference_generation_runtime)
        self.assertEqual(
            _ControlledReferenceThread.created[-1].prompt,
            DEFAULT_REFERENCE_VIEW_PRESETS[0].prompt,
        )

    def test_selected_preset_uses_edited_visible_prompt_in_request(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[1]
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(preset.preset_id))
        self.workspace.reference_generation_prompt_edit.setText("Use this edited prompt")
        self.workspace._activate_temporary_object_reference(
            _opaque_reference(), source_label="Loaded image"
        )
        self.workspace.generate_reference_button.click()
        self.assertEqual(
            _ControlledReferenceThread.created[-1].prompt, "Use this edited prompt"
        )

    def test_custom_prompt_is_forwarded_when_write_prompt_is_selected(self):
        thread = self._start(_opaque_reference())
        self.assertEqual(thread.prompt, "A plain wooden chair without the fabric cover")

    def test_selected_view_and_edited_prompt_persist_across_reopening(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[2]
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(preset.preset_id))
        self.workspace.reference_generation_prompt_edit.setText("Retained custom prompt")
        self.workspace.reference_generation_prompt_edit.editingFinished.emit()
        with self._reopened_workspace() as reopened:
            self.assertEqual(
                reopened.reference_generation_view_combo.currentData(), preset.preset_id
            )
            self.assertFalse(reopened.reference_generation_prompt_control.isHidden())
            self.assertEqual(
                reopened.reference_generation_prompt_edit.text(), "Retained custom prompt"
            )

    def test_reopening_loads_saved_preset_list_and_selection(self):
        preset = ReferenceViewPreset(
            preset_id="my-view", name="My saved view", prompt="A custom saved viewpoint"
        )
        save_reference_view_presets(self.settings, (preset,))
        self.settings.set(REFERENCE_GENERATION_VIEW_SETTINGS_KEY, preset.preset_id)
        with self._reopened_workspace() as reopened:
            combo = reopened.reference_generation_view_combo
            self.assertEqual(combo.count(), 2)
            self.assertEqual(combo.currentData(), preset.preset_id)
            self.assertEqual(combo.currentText(), preset.name)
            reopened._activate_temporary_object_reference(
                _opaque_reference(), source_label="Loaded image"
            )
            reopened.generate_reference_button.click()
            self.assertEqual(_ControlledReferenceThread.created[-1].prompt, preset.prompt)

    def test_saved_front_selection_survives_reordering_an_existing_preset_list(self):
        elevated, front, *other_views = DEFAULT_REFERENCE_VIEW_PRESETS
        save_reference_view_presets(self.settings, (front, *other_views, elevated))
        self.settings.set(REFERENCE_GENERATION_VIEW_SETTINGS_KEY, "front")

        with self._reopened_workspace() as reopened:
            combo = reopened.reference_generation_view_combo
            self.assertEqual(combo.itemData(0), "elevated-three-quarter")
            self.assertEqual(combo.itemData(1), "front")
            self.assertEqual(combo.currentData(), "front")

    def test_reopening_a_deleted_preset_falls_back_to_write_prompt(self):
        self.settings.set(REFERENCE_GENERATION_VIEW_SETTINGS_KEY, "removed-view")
        self.settings.set(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY, "My saved prompt")
        with self._reopened_workspace() as reopened:
            self.assertEqual(reopened.reference_generation_view_combo.currentData(), "")
            self.assertFalse(reopened.reference_generation_prompt_control.isHidden())
            self.assertEqual(
                reopened.reference_generation_prompt_edit.text(), "My saved prompt"
            )

    def test_reopening_removed_builtin_views_keeps_custom_prompt_and_remaining_views(self):
        retired_views = (
            ReferenceViewPreset("side", "Side/profile", "An old side view."),
            ReferenceViewPreset("top", "Top-down", "An old overhead view."),
        )
        save_reference_view_presets(
            self.settings, (*DEFAULT_REFERENCE_VIEW_PRESETS, *retired_views)
        )
        self.settings.set(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY, "My saved prompt")
        for removed_id in ("side", "top"):
            with self.subTest(view=removed_id):
                self.settings.set(REFERENCE_GENERATION_VIEW_SETTINGS_KEY, removed_id)
                with self._reopened_workspace() as reopened:
                    combo = reopened.reference_generation_view_combo
                    self.assertEqual(combo.count(), 5)
                    self.assertEqual(combo.findData("side"), -1)
                    self.assertEqual(combo.findData("top"), -1)
                    self.assertEqual(combo.currentData(), "")
                    self.assertFalse(reopened.reference_generation_prompt_control.isHidden())
                    self.assertEqual(
                        reopened.reference_generation_prompt_edit.text(), "My saved prompt"
                    )

    def test_live_preset_updates_keep_selection_and_use_updated_prompt(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[1]
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(preset.preset_id))
        replacement = ReferenceViewPreset(
            preset_id=preset.preset_id,
            name="Renamed view",
            prompt="A revised viewpoint description",
        )
        self.workspace.set_reference_view_presets((replacement,))
        self.assertEqual(combo.currentData(), preset.preset_id)
        self.assertEqual(combo.currentText(), "Renamed view")
        self.workspace._activate_temporary_object_reference(
            _opaque_reference(), source_label="Loaded image"
        )
        self.workspace.generate_reference_button.click()
        self.assertEqual(
            _ControlledReferenceThread.created[-1].prompt,
            "A revised viewpoint description",
        )

    def test_settings_view_edits_and_removals_update_generation_immediately(self):
        settings_widget = SettingsWidget(
            application_settings=self.settings, environment={}
        )
        try:
            settings_widget.reference_view_presets_changed.connect(
                self.workspace.set_reference_view_presets
            )
            combo = self.workspace.reference_generation_view_combo
            combo.setCurrentIndex(combo.findData("front"))
            settings_widget.reference_view_preset_list.setCurrentRow(1)
            self.assertEqual(combo.currentData(), "front")
            settings_widget.reference_view_name_edit.setText("Custom frontal view")
            settings_widget.reference_view_prompt_edit.setPlainText(
                "A clean frontal reference with the complete object visible"
            )
            settings_widget.save_reference_view_button.click()

            self.assertEqual(combo.currentData(), "front")
            self.assertEqual(combo.currentText(), "Custom frontal view")
            self.workspace._activate_temporary_object_reference(
                _opaque_reference(), source_label="Loaded image"
            )
            self.workspace.generate_reference_button.click()
            thread = _ControlledReferenceThread.created[-1]
            self.assertEqual(
                thread.prompt,
                "A clean frontal reference with the complete object visible",
            )
            thread.complete(_opaque_reference(190))

            for _ in range(settings_widget.reference_view_preset_list.count()):
                settings_widget.remove_reference_view_button.click()
            self.assertEqual(settings_widget.reference_view_preset_list.count(), 0)
            self.assertEqual(read_reference_view_presets(self.settings), ())
            self.assertEqual(combo.count(), 1)
            self.assertEqual(combo.currentText(), "(Write prompt)")
            self.assertEqual(combo.currentData(), "")
            self.assertFalse(self.workspace.reference_generation_prompt_control.isHidden())
            self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        finally:
            settings_widget.reference_view_presets_changed.disconnect(
                self.workspace.set_reference_view_presets
            )
            settings_widget.dispose()
            _delete_test_widgets(settings_widget)

    def test_removed_selected_preset_falls_back_to_custom_without_erasing_text(self):
        self.workspace.reference_generation_prompt_edit.setText("My retained prompt")
        self.workspace._activate_temporary_object_reference(
            _opaque_reference(), source_label="Loaded image"
        )
        self.workspace.set_reference_view_presets(DEFAULT_REFERENCE_VIEW_PRESETS[1:])
        self.assertEqual(self.workspace.reference_generation_view_combo.currentData(), "")
        self.assertFalse(self.workspace.reference_generation_prompt_control.isHidden())
        self.assertEqual(
            self.workspace.reference_generation_prompt_edit.text(), "My retained prompt"
        )
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        self.assertEqual(self.settings.get(REFERENCE_GENERATION_VIEW_SETTINGS_KEY), "")

    def test_empty_preset_list_keeps_write_prompt_available(self):
        self.workspace.set_reference_view_presets(())
        combo = self.workspace.reference_generation_view_combo
        self.assertEqual(combo.count(), 1)
        self.assertEqual(combo.currentText(), "(Write prompt)")
        self.assertEqual(combo.currentData(), "")
        self.assertFalse(self.workspace.reference_generation_prompt_control.isHidden())
        self.workspace.reference_generation_prompt_edit.clear()
        self.workspace._activate_temporary_object_reference(
            _opaque_reference(), source_label="Loaded image"
        )
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        self.workspace.reference_generation_prompt_edit.setText("A custom view")
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())

    def test_view_is_disabled_during_generation_and_prompt_remains_a_snapshot(self):
        preset = DEFAULT_REFERENCE_VIEW_PRESETS[0]
        self.workspace._activate_temporary_object_reference(
            _opaque_reference(), source_label="Loaded image"
        )
        self.workspace.generate_reference_button.click()
        thread = _ControlledReferenceThread.created[-1]
        self.assertEqual(thread.prompt, preset.prompt)
        self.assertFalse(self.workspace.reference_generation_view_combo.isEnabled())
        updated = ReferenceViewPreset(
            preset_id=preset.preset_id,
            name=preset.name,
            prompt="A prompt edited while generating",
        )
        self.workspace.set_reference_view_presets((updated,))
        self.assertEqual(thread.prompt, preset.prompt)
        self.assertFalse(self.workspace.reference_generation_view_combo.isEnabled())
        thread.complete(_opaque_reference(190))
        self.assertTrue(self.workspace.reference_generation_view_combo.isEnabled())
        self.workspace.generate_reference_button.click()
        self.assertEqual(
            _ControlledReferenceThread.created[-1].prompt,
            "A prompt edited while generating",
        )

    def test_reference_options_are_persisted_separately_from_meshy_prompt(self):
        self.workspace.ai_prompt_edit.setText("Meshy prompt")
        combo = self.settings_widget.reference_generation_model_combo
        combo.setCurrentIndex(
            combo.findData(REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION)
        )
        self.workspace.reference_generation_prompt_edit.setText("Reference prompt")
        view_combo = self.workspace.reference_generation_view_combo
        view_combo.setCurrentIndex(view_combo.findData(""))
        self.workspace.reference_generation_strength_spinbox.setValue(0.65)
        self.workspace.reference_generation_negative_prompt_edit.setText("blur, clutter")
        self.workspace.reference_generation_negative_prompt_edit.editingFinished.emit()
        self.assertEqual(
            self.settings.get(REFERENCE_GENERATION_MODEL_SETTINGS_KEY),
            REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
        )
        self.assertEqual(
            self.settings.get(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY),
            "Reference prompt",
        )
        self.assertEqual(
            self.settings.get(REFERENCE_GENERATION_VIEW_SETTINGS_KEY), ""
        )
        self.assertEqual(
            self.settings.get(REFERENCE_GENERATION_STRENGTH_SETTINGS_KEY), 0.65
        )
        self.assertEqual(
            self.settings.get(REFERENCE_GENERATION_NEGATIVE_PROMPT_SETTINGS_KEY),
            "blur, clutter",
        )
        self.assertEqual(self.workspace.ai_prompt_edit.text(), "Meshy prompt")

    def test_persisted_preferences_are_restored_on_reopening(self):
        self.settings.set(
            REFERENCE_GENERATION_MODEL_SETTINGS_KEY,
            REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
        )
        self.settings.set(REFERENCE_GENERATION_PROMPT_SETTINGS_KEY, "A bare table")
        self.settings.set(REFERENCE_GENERATION_VIEW_SETTINGS_KEY, "")
        self.settings.set(REFERENCE_GENERATION_STRENGTH_SETTINGS_KEY, 0.6)
        self.settings.set(
            REFERENCE_GENERATION_NEGATIVE_PROMPT_SETTINGS_KEY,
            "blurry, wall, floor",
        )
        other_objects = GenerationWorkspace(asset_directory=self.root / "other_objects")
        other_surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=self.root / "other_surfaces",
            application_settings=self.settings,
            shared_controls=other_objects.get_shared_controls(),
        )
        other = MergedGenerationWorkspace(
            other_surfaces,
            other_objects,
            GenerationJobManager(),
            application_settings=self.settings,
        )
        try:
            self.assertEqual(
                other._reference_generation_model,
                REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION,
            )
            self.assertEqual(
                other.reference_generation_prompt_edit.text(), "A bare table"
            )
            self.assertEqual(other.reference_generation_view_combo.currentData(), "")
            self.assertFalse(other.reference_generation_prompt_control.isHidden())
            self.assertEqual(other.reference_generation_strength_spinbox.value(), 0.6)
            self.assertFalse(other.reference_generation_strength_control.isHidden())
            self.assertFalse(
                other.reference_generation_negative_prompt_control.isHidden()
            )
            self.assertEqual(
                other.reference_generation_negative_prompt_edit.text(),
                "blurry, wall, floor",
            )
        finally:
            other.shutdown()
            other_objects.shutdown()
            other_surfaces.shutdown()
            _delete_test_widgets(other)

    def test_explicit_empty_negative_prompt_is_persisted_and_restored(self):
        self.workspace.reference_generation_negative_prompt_edit.clear()
        self.workspace.reference_generation_negative_prompt_edit.editingFinished.emit()
        self.assertEqual(
            self.settings.get(REFERENCE_GENERATION_NEGATIVE_PROMPT_SETTINGS_KEY), ""
        )
        reopened_settings = ApplicationSettingsStore(self.root / "settings.json")
        other_objects = GenerationWorkspace(asset_directory=self.root / "empty_objects")
        other_surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=self.root / "empty_surfaces",
            application_settings=reopened_settings,
            shared_controls=other_objects.get_shared_controls(),
        )
        other = MergedGenerationWorkspace(
            other_surfaces,
            other_objects,
            GenerationJobManager(),
            application_settings=reopened_settings,
        )
        try:
            self.assertEqual(other.reference_generation_negative_prompt_edit.text(), "")
        finally:
            other.shutdown()
            other_objects.shutdown()
            other_surfaces.shutdown()
            _delete_test_widgets(other)

    def test_negative_prompt_is_a_request_snapshot_and_disabled_while_running(self):
        self.workspace.reference_generation_negative_prompt_edit.setText("side view")
        thread = self._start(_opaque_reference())
        self.assertEqual(thread.negative_prompt, "side view")
        self.assertFalse(
            self.workspace.reference_generation_negative_prompt_edit.isEnabled()
        )
        self.workspace.reference_generation_negative_prompt_edit.setText("changed")
        self.assertEqual(thread.negative_prompt, "side view")
        thread.complete(_opaque_reference(190))
        self.assertTrue(
            self.workspace.reference_generation_negative_prompt_edit.isEnabled()
        )

    def test_empty_negative_prompt_is_passed_without_default_replacement(self):
        self.workspace.reference_generation_negative_prompt_edit.clear()
        thread = self._start(_opaque_reference())
        self.assertEqual(thread.negative_prompt, "")

    def test_sdxl_nonfront_preset_does_not_use_front_only_default_negatives(self):
        models = self.settings_widget.reference_generation_model_combo
        models.setCurrentIndex(models.findData(REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION))
        views = self.workspace.reference_generation_view_combo
        views.setCurrentIndex(views.findData("three-quarter"))
        negative_prompt = self.workspace.reference_generation_negative_prompt_edit.text()
        self.assertNotIn("side view", negative_prompt)
        self.assertNotIn("three-quarter view", negative_prompt)
        self.assertIn("blurry", negative_prompt)
        self.assertIn("mask outline", negative_prompt)
        self.workspace._activate_temporary_object_reference(
            _opaque_reference(), source_label="Loaded image"
        )
        self.workspace.generate_reference_button.click()
        self.assertEqual(
            _ControlledReferenceThread.created[-1].negative_prompt, negative_prompt
        )

    def test_sdxl_elevated_default_avoids_front_only_negatives_and_front_retains_them(self):
        models = self.settings_widget.reference_generation_model_combo
        models.setCurrentIndex(models.findData(REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION))
        views = self.workspace.reference_generation_view_combo
        self.assertEqual(views.currentData(), "elevated-three-quarter")
        negative_prompt = self.workspace.reference_generation_negative_prompt_edit.text()
        self.assertNotIn("side view", negative_prompt)
        self.assertNotIn("three-quarter view", negative_prompt)

        views.setCurrentIndex(views.findData("front"))
        self.workspace.reference_generation_negative_prompt_edit.setText(
            DEFAULT_REFERENCE_NEGATIVE_PROMPT
        )
        self.workspace._make_default_reference_negative_prompt_view_compatible()
        self.assertEqual(
            self.workspace.reference_generation_negative_prompt_edit.text(),
            DEFAULT_REFERENCE_NEGATIVE_PROMPT,
        )

    def test_switching_views_preserves_a_custom_negative_prompt(self):
        models = self.settings_widget.reference_generation_model_combo
        models.setCurrentIndex(models.findData(REFERENCE_GENERATION_MODEL_STABLE_DIFFUSION))
        negative_prompt = "side view, unique user-selected exclusions"
        self.workspace.reference_generation_negative_prompt_edit.setText(negative_prompt)
        views = self.workspace.reference_generation_view_combo
        views.setCurrentIndex(views.findData("three-quarter"))
        self.assertEqual(
            self.workspace.reference_generation_negative_prompt_edit.text(), negative_prompt
        )
        views.setCurrentIndex(views.findData("front"))
        self.assertEqual(
            self.workspace.reference_generation_negative_prompt_edit.text(), negative_prompt
        )

    def test_action_requires_input_and_a_nonempty_prompt(self):
        combo = self.workspace.reference_generation_view_combo
        combo.setCurrentIndex(combo.findData(""))
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        self.workspace.reference_generation_prompt_edit.setText("A chair")
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())
        self.workspace._activate_temporary_object_reference(
            _opaque_reference(), source_label="Loaded image"
        )
        self.assertTrue(self.workspace.generate_reference_button.isEnabled())
        self.workspace.reference_generation_prompt_edit.setText("   ")
        self.assertFalse(self.workspace.generate_reference_button.isEnabled())

    def test_mask_snapshot_is_sent_and_result_is_active_in_both_generators(self):
        self._load_video()
        original_mask = self.workspace.video_view.get_mask()
        original_frame = self.workspace.video_view.get_frame_bgr()
        expected = self.objects.get_current_object_generation_reference()
        thread = self._start()
        np.testing.assert_array_equal(thread.source_bgra, expected)
        self.assertEqual(self.workspace.generate_reference_button.text(), "Cancel")
        output = np.full((1024, 1024, 4), 190, dtype=np.uint8)
        output[:, :, 3] = 255
        thread.complete(output)
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), output
        )
        np.testing.assert_array_equal(self.surfaces.get_temporary_reference(), output)
        np.testing.assert_array_equal(
            self.workspace.video_view.get_reference_preview_bgra(), output
        )
        np.testing.assert_array_equal(
            self.workspace.video_view.get_mask(), original_mask
        )
        np.testing.assert_array_equal(
            self.workspace.video_view.get_frame_bgr(), original_frame
        )
        self.assertEqual(self.job_manager.jobs()[-1].status, "completed")
        self.assertEqual(
            self.workspace.generate_reference_button.text(), "Generate reference"
        )

    def test_retry_uses_original_source_not_previous_output(self):
        original = _opaque_reference(40)
        thread = self._start(original)
        output = _opaque_reference(190)
        thread.complete(output)
        second = self._start()
        np.testing.assert_array_equal(second.source_bgra, original)
        self.assertFalse(np.array_equal(second.source_bgra, output))

    def test_same_frame_refresh_preserves_generated_reference_and_retry_source(self):
        self._load_video()
        thread = self._start()
        original = thread.source_bgra.copy()
        output = _opaque_reference(190)
        thread.complete(output)
        self.objects.show_frame(0)
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), output
        )
        second = self._start()
        np.testing.assert_array_equal(second.source_bgra, original)

    def test_seek_restores_video_and_discards_pending_result(self):
        self._load_video()
        thread = self._start()
        self.workspace.seekbar.setValue(1)
        self.assertTrue(thread.interrupted)
        self.assertIsNone(self.workspace._reference_generation_runtime)
        thread.complete(_opaque_reference(190))
        self.assertFalse(self.objects.has_temporary_object_reference())
        self.assertFalse(self.workspace.video_view.has_reference_preview())
        self.assertEqual(self.job_manager.jobs()[-1].status, "cancelled")

    def test_mask_change_cancels_pending_result(self):
        self._load_video()
        thread = self._start()
        self.workspace.video_view.clear_mask()
        self.assertTrue(thread.interrupted)
        thread.complete(_opaque_reference(190))
        self.assertFalse(self.objects.has_temporary_object_reference())

    def test_new_pasted_input_cancels_old_request_and_does_not_get_overwritten(self):
        thread = self._start(_opaque_reference(30))
        new_image = QImage(20, 20, QImage.Format.Format_ARGB32)
        new_image.fill(0xFF2288AA)
        _qt_application.clipboard().setImage(new_image)
        self.workspace._handle_paste_inpaint_clicked()
        replacement = self.objects.get_current_object_generation_reference()
        self.assertTrue(thread.interrupted)
        thread.complete(_opaque_reference(190))
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), replacement
        )
        second = self._start()
        np.testing.assert_array_equal(second.source_bgra, replacement)

    def test_loading_image_cancels_pending_request_without_replacing_video(self):
        self._load_video()
        video_source = self.objects._video_source
        thread = self._start()
        image = QImage(30, 30, QImage.Format.Format_ARGB32)
        image.fill(0xFF112244)
        image_path = self.root / "loaded.png"
        self.assertTrue(image.save(str(image_path)))
        with patch(
            "housemaker.merged_generation_workspace.QFileDialog.getOpenFileName",
            return_value=(str(image_path), ""),
        ):
            self.workspace._handle_load_reference_image_clicked()
        replacement = self.objects.get_current_object_generation_reference()
        self.assertTrue(thread.interrupted)
        self.assertIs(self.objects._video_source, video_source)
        thread.complete(_opaque_reference(190))
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), replacement
        )

    def test_generated_reference_keeps_door_and_window_slot_targets(self):
        for asset_kind in ("door", "window"):
            with self.subTest(asset_kind=asset_kind):
                self.assertTrue(
                    self.objects.set_door_slot_editing_target(
                        door_id=f"{asset_kind}-one",
                        slot_id="body",
                        object_id=f"{asset_kind}-body",
                        object_name=f"New {asset_kind} 1",
                        slot_display_name="Body",
                        asset_kind=asset_kind,
                    )
                )
                target = self.objects._door_slot_editing_target
                thread = self._start(_opaque_reference())
                output = _opaque_reference(190)
                thread.complete(output)
                self.assertEqual(self.objects._door_slot_editing_target, target)
                np.testing.assert_array_equal(
                    self.objects.get_current_object_generation_reference(), output
                )

    def test_switching_door_slot_cancels_pending_result(self):
        self.assertTrue(
            self.objects.set_door_slot_editing_target(
                door_id="door-one",
                slot_id="body",
                object_id="body-one",
                object_name="New door 1",
                slot_display_name="Body",
            )
        )
        thread = self._start(_opaque_reference())
        self.assertTrue(
            self.objects.set_door_slot_editing_target(
                door_id="door-two",
                slot_id="body",
                object_id="body-two",
                object_name="New door 2",
                slot_display_name="Body",
            )
        )
        self.assertTrue(thread.interrupted)
        thread.complete(_opaque_reference(190))
        self.assertEqual(self.job_manager.jobs()[-1].status, "cancelled")

    def test_generated_reference_keeps_architectural_trim_target(self):
        self.objects._architectural_trim_editing_target = (
            _ArchitecturalTrimEditingTarget(
                level_index=2,
                trim_id="trim-one",
                object_id="cornice-one",
                object_name="Cornice",
                fit_dimensions=(2.0, 0.3, 0.3),
                trim_kind="cornice",
            )
        )
        self.objects.architectural_trim_editing_changed.emit(True)
        target = self.objects._architectural_trim_editing_target
        thread = self._start(_opaque_reference())
        output = _opaque_reference(190)
        thread.complete(output)
        self.assertEqual(self.objects._architectural_trim_editing_target, target)
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), output
        )

    def test_programmatic_input_replacement_is_guarded_at_completion(self):
        thread = self._start(_opaque_reference(30))
        replacement = _opaque_reference(90)
        self.objects.set_temporary_object_reference(replacement)
        thread.complete(_opaque_reference(190))
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), replacement
        )
        self.assertEqual(self.job_manager.jobs()[-1].status, "cancelled")

    def test_generate_button_cancels_without_losing_current_input(self):
        source = _opaque_reference()
        thread = self._start(source)
        self.workspace.generate_reference_button.click()
        self.assertTrue(thread.interrupted)
        self.assertEqual(self.job_manager.jobs()[-1].status, "cancelled")
        self.assertEqual(
            self.workspace.generate_reference_button.text(), "Generate reference"
        )
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), source
        )
        thread.complete(_opaque_reference(190))
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), source
        )

    def test_model_error_is_reported_without_a_fallback(self):
        source = _opaque_reference()
        thread = self._start(source)
        thread.complete(error="Model download failed: unavailable")
        self.assertIn(
            "Model download failed",
            self.workspace.reference_generation_status_label.text(),
        )
        self.assertEqual(self.job_manager.jobs()[-1].status, "failed")
        np.testing.assert_array_equal(
            self.objects.get_current_object_generation_reference(), source
        )

    def test_progress_updates_job_and_shutdown_cancels_request(self):
        thread = self._start(_opaque_reference())
        thread.progress_changed.emit("Downloading model", 30)
        self.assertEqual(self.job_manager.jobs()[-1].stage, "Downloading model")
        self.assertEqual(self.job_manager.jobs()[-1].progress, 30)
        self.workspace.shutdown()
        self.assertTrue(thread.interrupted)
        self.assertEqual(self.job_manager.jobs()[-1].status, "cancelled")
        self.assertIsNone(self.workspace._reference_generation_runtime)


# ### Reference generation worker tests ###
class ReferenceGenerationThreadTests(unittest.TestCase):
    def test_worker_lazily_calls_provider_with_snapshot_and_reports_progress(self):
        source = _opaque_reference()
        output = _opaque_reference(190)
        calls = []

        def generate(source_bgra, prompt, model, **options):
            calls.append((source_bgra.copy(), prompt, model, options))
            options["progress_callback"]("Generating", 60)
            self.assertFalse(options["cancellation_check"]())
            return output

        backend = ModuleType("housemaker.reference_generation")
        backend.generate_object_reference = generate
        thread = _ReferenceGenerationThread(
            source,
            "A bare chair",
            "test-model",
            api_key="secret",
            strength=0.6,
            negative_prompt="side view, clutter",
        )
        progress = []
        thread.progress_changed.connect(lambda *values: progress.append(values))
        source[:, :, :3] = 3
        with patch.dict(sys.modules, {"housemaker.reference_generation": backend}):
            thread.run()
        self.assertIs(thread.result, output)
        self.assertIsNone(thread.error_message)
        self.assertEqual(calls[0][1:3], ("A bare chair", "test-model"))
        self.assertEqual(calls[0][3]["strength"], 0.6)
        self.assertEqual(calls[0][3]["negative_prompt"], "side view, clutter")
        self.assertEqual(calls[0][3]["api_key"], "secret")
        self.assertTrue(np.all(calls[0][0][:, :, 0] == 70))
        self.assertEqual(progress, [("Generating", 60)])
        self.assertEqual(thread._api_key, "")

    def test_worker_redacts_api_key_from_provider_errors(self):
        backend = ModuleType("housemaker.reference_generation")

        def generate(*_args, **_options):
            raise RuntimeError("Provider rejected secret-key")

        backend.generate_object_reference = generate
        thread = _ReferenceGenerationThread(
            _opaque_reference(),
            "A chair",
            "test-model",
            api_key="secret-key",
            strength=0.45,
        )
        with patch.dict(sys.modules, {"housemaker.reference_generation": backend}):
            thread.run()
        self.assertEqual(thread.error_message, "Provider rejected [redacted]")
        self.assertIsNone(thread.result)
        self.assertFalse(thread.was_cancelled)


# ### Test entry point ###
if __name__ == "__main__":
    unittest.main()
