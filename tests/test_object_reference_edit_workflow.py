# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QPushButton

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.merged_generation_workspace import (
    ARCHITECTURAL_TRIM_REFERENCE_EDIT_INSTRUCTION,
    OBJECT_REFERENCE_EDIT_JOB_KIND,
    MergedGenerationWorkspace,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_texture_workspace import (
    SurfaceTextureGenerationWorkspace,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixture helpers ###
def _stroke(
    x_position: float,
    *,
    y_position: float = 0.5,
    radius: float = 0.1,
) -> MaskStroke:
    return MaskStroke(
        MASK_MODE_PAINT,
        radius,
        (MaskPoint(x_position, y_position),),
    )


def _reference_frame() -> np.ndarray:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :]
    frame[:, :, 1] = np.arange(48, dtype=np.uint8)[:, None]
    frame[:, :, 2] = 180
    return frame


def _temporary_reference() -> np.ndarray:
    reference = np.empty((37, 53, 4), dtype=np.uint8)
    reference[:, :, 0] = np.arange(53, dtype=np.uint8)[None, :]
    reference[:, :, 1] = np.arange(37, dtype=np.uint8)[:, None]
    reference[:, :, 2] = 210
    reference[:, :, 3] = 255
    reference[:5, :7, 3] = 0
    return reference


def _write_test_video(path: Path, *, frame_count: int = 1) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("MJPG video writing is unavailable.")
    try:
        for frame_index in range(frame_count):
            frame = _reference_frame()
            frame[:, :, 2] = max(0, 180 - (frame_index * 40))
            writer.write(frame)
    finally:
        writer.release()


def _wait_until(predicate, timeout_seconds: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _qt_application.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    _qt_application.processEvents()
    return bool(predicate())


# ### Generation-workspace tests ###
class ObjectReferenceEditGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = GenerationWorkspace(
            asset_directory=Path(self.temporary_directory.name) / "objects",
        )
        self.frame = _reference_frame()
        self.object_strokes = [_stroke(0.5, radius=0.28)]
        self.workspace.video_view.set_frame(self.frame, self.object_strokes)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def _accept_distinct_edit(self) -> np.ndarray:
        _source, signature = (
            self.workspace.build_object_reference_edit_inputs()
        )
        edited = np.empty((1024, 1024, 4), dtype=np.uint8)
        edited[:] = (231, 77, 19, 255)
        self.assertTrue(self.workspace.accept_object_reference_edit(signature, edited))
        return edited

    def test_accepted_edit_substitutes_both_generation_modes_non_destructively(
        self,
    ) -> None:
        original_frame = self.workspace.video_view.get_frame_bgr()
        original_object_strokes = self.workspace.video_view.get_strokes()
        original_object_mask = self.workspace.video_view.get_mask()
        edited = self._accept_distinct_edit()

        textured_requests = self.workspace._build_generation_requests()
        geometry_requests = self.workspace._build_generation_requests(
            geometry_only=True
        )

        self.assertEqual(len(textured_requests), 1)
        self.assertEqual(len(geometry_requests), 1)
        self.assertFalse(textured_requests[0].geometry_only)
        self.assertTrue(geometry_requests[0].geometry_only)
        np.testing.assert_array_equal(
            textured_requests[0].selected_object_bgra,
            edited,
        )
        np.testing.assert_array_equal(
            geometry_requests[0].selected_object_bgra,
            edited,
        )
        np.testing.assert_array_equal(
            self.workspace.video_view.get_frame_bgr(),
            original_frame,
        )
        np.testing.assert_array_equal(
            self.workspace.video_view.get_mask(),
            original_object_mask,
        )
        self.assertEqual(
            self.workspace.video_view.get_strokes(),
            original_object_strokes,
        )

    def test_accepted_edit_is_sole_input_when_mask_changes_or_clears(self) -> None:
        edited = self._accept_distinct_edit()
        self.assertTrue(self.workspace.has_current_accepted_object_reference())

        self.workspace.video_view.set_strokes(
            [_stroke(0.25, radius=0.08), _stroke(0.75, radius=0.08)]
        )

        self.assertTrue(self.workspace.has_current_accepted_object_reference())
        requests = self.workspace._build_generation_requests()
        self.assertEqual(len(requests), 1)
        np.testing.assert_array_equal(requests[0].selected_object_bgra, edited)

        self.workspace.video_view.set_strokes([])

        self.assertTrue(self.workspace.has_current_accepted_object_reference())
        requests = self.workspace._build_generation_requests()
        self.assertEqual(len(requests), 1)
        np.testing.assert_array_equal(requests[0].selected_object_bgra, edited)

    def test_leaving_frame_permanently_evicts_accepted_edit(self) -> None:
        video_path = Path(self.temporary_directory.name) / "frame-reference.avi"
        _write_test_video(video_path, frame_count=2)
        self.workspace.load_video(str(video_path))
        self.workspace.video_view.set_strokes(self.object_strokes)
        self.workspace.video_view.strokes_changed.emit(self.object_strokes)
        self._accept_distinct_edit()
        self.assertTrue(self.workspace.has_current_accepted_object_reference())

        self.workspace.show_frame(1)
        self.assertFalse(self.workspace.has_current_accepted_object_reference())

        self.workspace.show_frame(0)

        self.assertFalse(self.workspace.has_current_accepted_object_reference())
        self.assertIsNone(self.workspace.get_current_accepted_object_reference())


# ### Merged-workspace async workflow tests ###
class ObjectReferenceEditUiWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.job_manager = GenerationJobManager()
        self.objects = GenerationWorkspace(
            asset_directory=root / "objects",
            job_manager=self.job_manager,
        )
        self.surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=root / "surfaces",
            application_settings=ApplicationSettingsStore(root / "settings.json"),
            job_manager=self.job_manager,
            shared_controls=self.objects.get_shared_controls(),
        )
        self.editor_calls: list[tuple[np.ndarray, str, str, str]] = []

        def editor(
            source_bgra: np.ndarray,
            prompt: str,
            *,
            model: str,
            api_key: str,
            cancellation_check=None,
        ) -> np.ndarray:
            call_index = len(self.editor_calls)
            self.editor_calls.append(
                (
                    source_bgra.copy(),
                    prompt,
                    model,
                    api_key,
                )
            )
            result = np.empty((1024, 1024, 4), dtype=np.uint8)
            result[:] = (9 + call_index, 171, 244, 255)
            return result

        self.workspace = MergedGenerationWorkspace(
            self.surfaces,
            self.objects,
            self.job_manager,
            object_reference_editor=editor,
        )
        self.objects.set_runtime_settings(
            GenerationServiceSettings(openai_api_key="sk-test")
        )
        video_path = root / "reference.avi"
        _write_test_video(video_path, frame_count=2)
        self.objects.load_video(str(video_path))
        self.workspace.video_view.set_strokes([_stroke(0.5, radius=0.28)])
        self.workspace.reference_edit_prompt.setText(
            "Remove the floral tablecloth and reconstruct the bare tabletop."
        )
        self.workspace.sync_shared_controls()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.surfaces.shutdown()
        self.objects.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_edit_auto_accepts_and_retry_replaces_the_active_reference(
        self,
    ) -> None:
        original_frame = self.workspace.video_view.get_frame_bgr()
        original_object_strokes = self.workspace.video_view.get_strokes()
        original_object_mask = self.workspace.video_view.get_mask()
        self.assertFalse(self.workspace.edit_reference_button.isCheckable())
        self.assertTrue(self.workspace.edit_reference_button.isEnabled())
        self.workspace.reference_edit_model_combo.setCurrentIndex(
            self.workspace.reference_edit_model_combo.findData(
                "gpt-image-2.5-sunburst"
            )
        )

        self.workspace.edit_reference_button.click()

        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )
        self.assertEqual(len(self.editor_calls), 1)
        source_bgra, prompt, model, api_key = self.editor_calls[0]
        self.assertGreater(np.count_nonzero(source_bgra[:, :, 3]), 0)
        self.assertEqual(
            (prompt, model, api_key),
            (
                "Remove the floral tablecloth and reconstruct the bare tabletop.",
                "gpt-image-2.5-sunburst",
                "sk-test",
            ),
        )
        self.assertTrue(self.workspace.video_view.has_reference_preview())
        self.assertTrue(self.objects.has_current_accepted_object_reference())
        jobs = [
            job
            for job in self.job_manager.jobs()
            if job.kind == OBJECT_REFERENCE_EDIT_JOB_KIND
        ]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, "completed")

        preview = self.workspace.video_view.get_reference_preview_bgra()
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(preview.shape, (1024, 1024, 4))
        request = self.objects._build_generation_requests()[0]
        np.testing.assert_array_equal(request.selected_object_bgra, preview)
        np.testing.assert_array_equal(
            self.workspace.video_view.get_frame_bgr(),
            original_frame,
        )
        np.testing.assert_array_equal(
            self.workspace.video_view.get_mask(),
            original_object_mask,
        )
        self.assertEqual(
            self.workspace.video_view.get_strokes(),
            original_object_strokes,
        )
        self.assertFalse(self.workspace.paint_mask_button.isEnabled())
        self.assertFalse(self.workspace.erase_mask_button.isEnabled())
        self.assertFalse(self.workspace.clear_mask_button.isEnabled())

        self.workspace.edit_reference_button.click()
        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )

        self.assertEqual(len(self.editor_calls), 2)
        np.testing.assert_array_equal(
            self.editor_calls[1][0],
            self.editor_calls[0][0],
        )
        self.assertTrue(self.objects.has_current_accepted_object_reference())
        self.assertTrue(self.workspace.video_view.has_reference_preview())
        retried_preview = self.workspace.video_view.get_reference_preview_bgra()
        self.assertIsNotNone(retried_preview)
        assert retried_preview is not None
        self.assertFalse(np.array_equal(retried_preview, preview))
        retried_request = self.objects._build_generation_requests()[0]
        np.testing.assert_array_equal(
            retried_request.selected_object_bgra,
            retried_preview,
        )

    def test_temporary_reference_can_be_edited_and_retried_from_its_source(
        self,
    ) -> None:
        source = _temporary_reference()
        self.workspace._activate_temporary_object_reference(
            source,
            source_label="Loaded image",
        )

        self.assertTrue(self.workspace.edit_reference_button.isEnabled())
        self.workspace.edit_reference_button.click()
        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )

        self.assertEqual(len(self.editor_calls), 1)
        np.testing.assert_array_equal(self.editor_calls[0][0], source)
        self.assertTrue(self.objects.has_temporary_object_reference())
        self.assertFalse(self.objects.has_current_accepted_object_reference())
        first_result = self.objects.get_temporary_object_reference()
        self.assertIsNotNone(first_result)
        np.testing.assert_array_equal(
            self.workspace.video_view.get_reference_preview_bgra(),
            first_result,
        )
        np.testing.assert_array_equal(
            self.objects._build_generation_requests()[0].selected_object_bgra,
            first_result,
        )

        self.workspace.edit_reference_button.click()
        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )

        self.assertEqual(len(self.editor_calls), 2)
        np.testing.assert_array_equal(self.editor_calls[1][0], source)
        second_result = self.objects.get_temporary_object_reference()
        self.assertIsNotNone(second_result)
        self.assertFalse(np.array_equal(first_result, second_result))
        np.testing.assert_array_equal(
            self.workspace.video_view.get_reference_preview_bgra(),
            second_result,
        )

    def test_trim_selection_prefills_only_an_empty_edit_instruction(self) -> None:
        self.workspace.reference_edit_prompt.clear()

        self.objects.architectural_trim_editing_changed.emit(True)

        self.assertEqual(
            self.workspace.reference_edit_prompt.text(),
            ARCHITECTURAL_TRIM_REFERENCE_EDIT_INSTRUCTION,
        )

        custom_prompt = "Keep this custom cornice instruction."
        self.workspace.reference_edit_prompt.setText(custom_prompt)
        self.objects.architectural_trim_editing_changed.emit(True)

        self.assertEqual(self.workspace.reference_edit_prompt.text(), custom_prompt)

    def test_manual_reference_edit_buttons_are_removed(self) -> None:
        old_button_names = (
            "accept_object_reference_edit_button",
            "retry_object_reference_edit_button",
            "undo_object_reference_edit_button",
        )
        for object_name in old_button_names:
            with self.subTest(object_name=object_name):
                self.assertIsNone(
                    self.workspace.findChild(QPushButton, object_name)
                )
        self.assertFalse(hasattr(self.workspace, "accept_reference_edit_button"))
        self.assertFalse(hasattr(self.workspace, "retry_reference_edit_button"))
        self.assertFalse(hasattr(self.workspace, "undo_reference_edit_button"))

    def test_merged_seekbar_frame_change_permanently_discards_accepted_edit(
        self,
    ) -> None:
        self.workspace.edit_reference_button.click()
        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )
        self.assertTrue(self.objects.has_current_accepted_object_reference())
        self.assertTrue(self.workspace.video_view.has_reference_preview())

        self.workspace.seekbar.setValue(0)
        _qt_application.processEvents()

        self.assertTrue(self.objects.has_current_accepted_object_reference())
        self.assertTrue(self.workspace.video_view.has_reference_preview())

        self.workspace.seekbar.setValue(1)
        _qt_application.processEvents()

        self.assertEqual(self.objects.get_data().current_frame_index, 1)
        self.assertFalse(self.objects.has_current_accepted_object_reference())
        self.assertFalse(self.workspace.video_view.has_reference_preview())
        self.assertTrue(self.workspace.paint_mask_button.isEnabled())

        self.workspace.seekbar.setValue(0)
        _qt_application.processEvents()

        self.assertEqual(self.objects.get_data().current_frame_index, 0)
        self.assertFalse(self.objects.has_current_accepted_object_reference())
        self.assertIsNone(self.objects.get_current_accepted_object_reference())

    def test_model_selector_exists_and_defaults_to_current_model(self) -> None:
        self.assertEqual(
            self.workspace.reference_edit_model_combo.currentData(),
            "gpt-image-2",
        )
        actual_options = {
            self.workspace.reference_edit_model_combo.itemText(index):
                self.workspace.reference_edit_model_combo.itemData(index)
            for index in range(
                self.workspace.reference_edit_model_combo.count()
            )
        }
        self.assertEqual(
            actual_options,
            {
                "GPT Image 2.5 Sunburst": "gpt-image-2.5-sunburst",
                "GPT Image 2.5 Flare": "gpt-image-2.5-flare",
                "GPT Image 2": "gpt-image-2",
                "Qwen-Image-2.1": "Qwen-Image-2.1",
                "GPT-5.6 Terra": "gpt-5.6-terra",
                "GPT-5.6 Luna": "gpt-5.6-luna",
            },
        )

    def test_view_preset_is_above_edit_instruction(self) -> None:
        layout = self.workspace.reference_editing_section.layout()
        preset_wrapper = (
            self.workspace.reference_edit_prompt_preset_combo.parentWidget()
        )
        prompt_wrapper = self.workspace.reference_edit_prompt.parentWidget()

        self.assertGreaterEqual(layout.indexOf(preset_wrapper), 0)
        self.assertGreaterEqual(layout.indexOf(prompt_wrapper), 0)
        self.assertLess(
            layout.indexOf(preset_wrapper),
            layout.indexOf(prompt_wrapper),
        )

    def test_view_prompt_presets_append_with_commas_and_reset(self) -> None:
        combo = self.workspace.reference_edit_prompt_preset_combo
        expected_presets = (
            "show me a front of it",
            "slightly rotated to the left",
            "slightly rotated to the right",
            "slightly isometric",
            "top-down view",
        )
        self.assertEqual(combo.count(), len(expected_presets) + 1)
        self.assertIsNone(combo.itemData(0))
        self.assertEqual(
            tuple(combo.itemData(index) for index in range(1, combo.count())),
            expected_presets,
        )
        self.assertEqual(
            tuple(combo.itemText(index) for index in range(1, combo.count())),
            expected_presets,
        )

        self.workspace.reference_edit_prompt.setText("Remove the cover")
        left_index = combo.findData("slightly rotated to the left")
        combo.setCurrentIndex(left_index)
        self.assertEqual(
            self.workspace.reference_edit_prompt.text(),
            "Remove the cover",
        )
        combo.activated.emit(left_index)

        self.assertEqual(
            self.workspace.reference_edit_prompt.text(),
            "Remove the cover, slightly rotated to the left",
        )
        self.assertEqual(combo.currentIndex(), 0)

        right_index = combo.findData("slightly rotated to the right")
        combo.setCurrentIndex(right_index)
        combo.activated.emit(right_index)
        self.assertEqual(
            self.workspace.reference_edit_prompt.text(),
            "Remove the cover, slightly rotated to the left, slightly "
            "rotated to the right",
        )

        self.workspace.reference_edit_prompt.setText("Rotate the object, ")
        top_down_index = combo.findData("top-down view")
        combo.setCurrentIndex(top_down_index)
        combo.activated.emit(top_down_index)
        self.assertEqual(
            self.workspace.reference_edit_prompt.text(),
            "Rotate the object, top-down view",
        )

        self.workspace.reference_edit_prompt.clear()
        front_index = combo.findData("show me a front of it")
        combo.setCurrentIndex(front_index)
        combo.activated.emit(front_index)
        self.assertEqual(
            self.workspace.reference_edit_prompt.text(),
            "show me a front of it",
        )
        self.workspace.edit_reference_button.click()
        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )
        self.assertEqual(self.editor_calls[-1][1], "show me a front of it")

    def test_view_prompt_preset_rejects_an_overlong_append_atomically(
        self,
    ) -> None:
        combo = self.workspace.reference_edit_prompt_preset_combo
        preset = "top-down view"
        preset_index = combo.findData(preset)
        maximum_length = self.workspace.reference_edit_prompt.maxLength()
        exact_prefix = "x" * (maximum_length - len(", ") - len(preset))
        self.workspace.reference_edit_prompt.setText(exact_prefix)

        combo.setCurrentIndex(preset_index)
        combo.activated.emit(preset_index)

        self.assertEqual(
            len(self.workspace.reference_edit_prompt.text()),
            maximum_length,
        )
        self.assertTrue(
            self.workspace.reference_edit_prompt.text().endswith(
                f", {preset}"
            )
        )

        overlong_prefix = exact_prefix + "x"
        self.workspace.reference_edit_prompt.setText(overlong_prefix)
        combo.setCurrentIndex(preset_index)
        combo.activated.emit(preset_index)

        self.assertEqual(
            self.workspace.reference_edit_prompt.text(),
            overlong_prefix,
        )
        self.assertEqual(combo.currentIndex(), 0)
        self.assertIn(
            "exceed the edit instruction limit",
            self.workspace.reference_edit_status_label.text(),
        )

    def test_each_new_model_is_propagated_to_the_reference_editor(self) -> None:
        for model in ("Qwen-Image-2.1", "gpt-5.6-terra", "gpt-5.6-luna"):
            with self.subTest(model=model):
                model_index = self.workspace.reference_edit_model_combo.findData(
                    model
                )
                self.assertGreaterEqual(model_index, 0)
                self.workspace.reference_edit_model_combo.setCurrentIndex(
                    model_index
                )
                self.workspace.edit_reference_button.click()
                self.assertTrue(
                    _wait_until(
                        lambda: self.workspace._object_reference_edit_runtime
                        is None
                    )
                )
                self.assertEqual(self.editor_calls[-1][2], model)

    def test_local_qwen_edit_is_enabled_and_runs_without_openai_key(self) -> None:
        self.objects.set_runtime_settings(GenerationServiceSettings())
        qwen_index = self.workspace.reference_edit_model_combo.findData(
            "Qwen-Image-2.1"
        )
        self.assertGreaterEqual(qwen_index, 0)
        self.workspace.reference_edit_model_combo.setCurrentIndex(qwen_index)
        self.workspace.sync_shared_controls()

        self.assertTrue(self.workspace.edit_reference_button.isEnabled())
        self.workspace.edit_reference_button.click()

        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )
        self.assertEqual(len(self.editor_calls), 1)
        self.assertEqual(self.editor_calls[0][2:], ("Qwen-Image-2.1", ""))
        self.assertTrue(self.workspace.video_view.has_reference_preview())
        self.assertTrue(self.objects.has_current_accepted_object_reference())

    def test_remote_models_still_require_an_openai_key(self) -> None:
        self.objects.set_runtime_settings(GenerationServiceSettings())
        for model in (
            "gpt-image-2",
            "gpt-image-2.5-sunburst",
            "gpt-image-2.5-flare",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
        ):
            with self.subTest(model=model):
                model_index = self.workspace.reference_edit_model_combo.findData(
                    model
                )
                self.assertGreaterEqual(model_index, 0)
                self.workspace.reference_edit_model_combo.setCurrentIndex(
                    model_index
                )
                self.workspace.sync_shared_controls()
                self.assertFalse(self.workspace.edit_reference_button.isEnabled())

    def test_failed_retry_preserves_the_previous_accepted_reference(self) -> None:
        self.workspace.edit_reference_button.click()
        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )
        accepted = self.objects.get_current_accepted_object_reference()
        preview = self.workspace.video_view.get_reference_preview_bgra()
        self.assertIsNotNone(accepted)
        self.assertIsNotNone(preview)

        self.workspace._object_reference_editor = (
            lambda *_args, **_kwargs: np.zeros((8, 8, 4), dtype=np.uint8)
        )
        self.workspace.edit_reference_button.click()
        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )

        self.assertTrue(self.objects.has_current_accepted_object_reference())
        np.testing.assert_array_equal(
            self.objects.get_current_accepted_object_reference(),
            accepted,
        )
        np.testing.assert_array_equal(
            self.workspace.video_view.get_reference_preview_bgra(),
            preview,
        )

    def test_malformed_editor_result_fails_job_without_preview_or_runtime(
        self,
    ) -> None:
        malformed_results = (
            np.zeros((1, 1, 4), dtype=np.uint8),
            np.zeros((48, 64, 4), dtype=np.float32),
        )
        for index, malformed_result in enumerate(malformed_results):
            with self.subTest(index=index):
                self.workspace._object_reference_editor = (
                    lambda *_args, result=malformed_result, **_kwargs: result
                )

                self.workspace.edit_reference_button.click()

                self.assertTrue(
                    _wait_until(
                        lambda: self.workspace._object_reference_edit_runtime is None
                    )
                )
                jobs = [
                    job
                    for job in self.job_manager.jobs()
                    if job.kind == OBJECT_REFERENCE_EDIT_JOB_KIND
                ]
                self.assertEqual(jobs[-1].status, "failed")
                self.assertIn("invalid image", jobs[-1].stage.lower())
                self.assertIsNone(self.workspace._object_reference_edit_runtime)
                self.assertFalse(self.workspace.video_view.has_reference_preview())
                self.assertFalse(
                    self.objects.has_current_accepted_object_reference()
                )

    def test_instruction_change_preserves_the_accepted_reference(self) -> None:
        self.workspace.edit_reference_button.click()
        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )
        self.assertTrue(self.workspace.video_view.has_reference_preview())
        accepted = self.objects.get_current_accepted_object_reference()
        self.assertIsNotNone(accepted)

        self.workspace.reference_edit_prompt.setText(
            "Replace the tabletop with plain unfinished oak."
        )
        _qt_application.processEvents()

        self.assertTrue(self.workspace.video_view.has_reference_preview())
        self.assertTrue(self.objects.has_current_accepted_object_reference())
        np.testing.assert_array_equal(
            self.objects.get_current_accepted_object_reference(),
            accepted,
        )

    def test_model_change_preserves_the_accepted_reference(self) -> None:
        self.workspace.edit_reference_button.click()
        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )
        self.assertTrue(self.workspace.video_view.has_reference_preview())
        accepted = self.objects.get_current_accepted_object_reference()
        self.assertIsNotNone(accepted)

        self.workspace.reference_edit_model_combo.setCurrentIndex(
            self.workspace.reference_edit_model_combo.findData(
                "gpt-image-2.5-flare"
            )
        )
        _qt_application.processEvents()

        self.assertTrue(self.workspace.video_view.has_reference_preview())
        self.assertTrue(self.objects.has_current_accepted_object_reference())
        np.testing.assert_array_equal(
            self.objects.get_current_accepted_object_reference(),
            accepted,
        )

    def test_object_mask_change_cancels_and_ignores_running_reference_edit(
        self,
    ) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocking_editor(
            source_bgra: np.ndarray,
            _prompt: str,
            *,
            model: str,
            api_key: str,
            cancellation_check=None,
        ) -> np.ndarray:
            self.assertEqual(model, "gpt-image-2")
            self.assertEqual(api_key, "sk-test")
            started.set()
            release.wait(timeout=2.0)
            return source_bgra.copy()

        self.workspace._object_reference_editor = blocking_editor
        try:
            self.workspace.edit_reference_button.click()
            self.assertTrue(started.wait(timeout=1.0))
            changed_strokes = [_stroke(0.4, radius=0.24)]
            self.workspace.video_view.set_strokes(changed_strokes)
            self.workspace.video_view.strokes_changed.emit(changed_strokes)
        finally:
            release.set()

        self.assertTrue(
            _wait_until(lambda: self.workspace._object_reference_edit_runtime is None)
        )
        jobs = [
            job
            for job in self.job_manager.jobs()
            if job.kind == OBJECT_REFERENCE_EDIT_JOB_KIND
        ]
        self.assertEqual(jobs[-1].status, "cancelled")
        self.assertFalse(self.workspace.video_view.has_reference_preview())
        self.assertFalse(self.objects.has_current_accepted_object_reference())


if __name__ == "__main__":
    unittest.main()
