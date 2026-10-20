"""Manual cross-frame object navigation through the shared Generation UI."""

# ### Environment and imports ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_state import MASK_MODE_PAINT, MaskPoint, MaskStroke
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.merged_generation_workspace import MergedGenerationWorkspace
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace

# ### Local video and controlled reference generation ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _stroke(x):
    return MaskStroke(MASK_MODE_PAINT, 0.07, (MaskPoint(x, 0.5),))


def _reference(value):
    image = np.full((20, 24, 4), value, np.uint8)
    image[:, :, 3] = 255
    return image


class _ReferenceThread(QObject):
    progress_changed = Signal(str, int)
    finished = Signal()

    def __init__(self, source, *_args, parent=None, **_kwargs):
        super().__init__(parent)
        self.source = source.copy()
        self.result = None
        self.error_message = None
        self.was_cancelled = False
        self.interrupted = False

    def start(self):
        pass

    def requestInterruption(self):
        self.interrupted = True

    def isRunning(self):
        return False

    def complete(self, image):
        self.result = image
        self.finished.emit()


# ### Shared UI integration tests (no AI or paid requests) ###
class ManualInpaintWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manager = GenerationJobManager()
        self.objects = GenerationWorkspace(asset_directory=self.root / "objects", job_manager=self.manager)
        self.surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=self.root / "surfaces", job_manager=self.manager,
            shared_controls=self.objects.get_shared_controls(),
        )
        self.workspace = MergedGenerationWorkspace(self.surfaces, self.objects, self.manager)
        self.objects.set_runtime_settings(GenerationServiceSettings(meshy_api_key="fake-test-key"))
        self.video = self.root / "manual.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        try:
            for value in (40, 80, 120):
                writer.write(np.full((48, 64, 3), value, np.uint8))
        finally:
            writer.release()
        self.objects.load_video(str(self.video))
        self.surfaces.load_video(str(self.video))
        self._paint(0, 0.25, 0.75)
        self._paint(2, 0.5)
        self.workspace.seekbar.setValue(0)
        self.workspace.video_view.select_blob(0)
        self.workspace.sync_shared_controls()

    def tearDown(self):
        self.workspace.shutdown()
        self.objects.shutdown()
        self.surfaces.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        APP.processEvents()
        self.temporary.cleanup()

    def _paint(self, frame, *positions):
        self.workspace.seekbar.setValue(frame)
        self.workspace.video_view.set_strokes([_stroke(x) for x in positions])
        self.workspace.video_view.strokes_changed.emit(self.workspace.video_view.get_strokes())

    def test_navigation_sits_above_division_and_wraps_across_frames(self):
        layout = self.workspace.object_creation_section.layout()
        settings = self.workspace.division_options_section.parentWidget()
        self.assertLess(layout.indexOf(self.workspace.inpaint_navigation_control), layout.indexOf(settings))
        self.assertEqual(self.workspace.inpaint_position_label.text(), "Inpaint 1 of 3")
        self.workspace.previous_inpaint_button.click()
        self.assertEqual(self.objects._displayed_frame_index, 2)
        self.assertEqual(self.surfaces._displayed_frame_index, 2)
        self.assertEqual(self.workspace.inpaint_position_label.text(), "Inpaint 3 of 3")
        self.workspace.next_inpaint_button.click()
        self.assertEqual(self.objects._displayed_frame_index, 0)
        self.assertEqual(self.workspace.video_view.get_selected_blob_index(), 0)
        self.workspace.next_inpaint_button.click()
        self.assertEqual(self.workspace.video_view.get_selected_blob_index(), 1)

    def test_external_references_stay_with_each_object_across_frames(self):
        for value in (11, 22, 33):
            self.workspace._activate_temporary_object_reference(_reference(value), source_label="Pasted test")
            self.workspace.next_inpaint_button.click()
        for value in (11, 22, 33):
            np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), _reference(value))
            np.testing.assert_array_equal(self.surfaces.get_temporary_reference(), _reference(value))
            self.workspace.next_inpaint_button.click()
        requests = self.objects._build_generation_requests()
        self.assertEqual([request.frame_index for request in requests], [0, 0, 2])
        self.assertEqual([int(request.selected_object_bgra[0, 0, 0]) for request in requests], [11, 22, 33])

    def test_clear_selected_blob_also_works_while_reference_preview_is_showing(self):
        self.workspace._activate_temporary_object_reference(_reference(11), source_label="Pasted test")
        self.assertTrue(self.workspace.clear_mask_button.isEnabled())
        self.workspace.clear_mask_button.click()
        self.assertEqual(self.objects.inpaint_queue(), ((0, 0), (2, 0)))
        self.assertIsNone(self.objects.get_temporary_object_reference())
        self.assertFalse(self.workspace.video_view.has_reference_preview())

    def test_unpainted_frame_does_not_disable_or_drop_other_frames_objects(self):
        self.workspace.seekbar.setValue(1)
        self.assertFalse(self.workspace.video_view.has_selection())
        self.assertTrue(self.objects.generate_button.isEnabled())
        self.assertTrue(self.objects.generate_geometry_button.isEnabled())
        requests = self.objects._build_generation_requests(geometry_only=True)
        self.assertEqual([request.frame_index for request in requests], [0, 0, 2])
        self.assertTrue(all(request.geometry_only for request in requests))
        self.assertEqual(self.objects._displayed_frame_index, 1)

    def test_four_objects_on_one_frame_and_two_on_another_start_six_jobs(self):
        self._paint(0, 0.12, 0.37, 0.62, 0.87)
        self._paint(2, 0.25, 0.75)
        with patch.object(self.objects, "_start_generation", side_effect=[f"test-{i}" for i in range(6)]) as start:
            self.objects.generate_geometry_button.click()
        self.assertEqual(start.call_count, 6)
        self.assertEqual([call.args[0].frame_index for call in start.call_args_list], [0, 0, 0, 0, 2, 2])
        self.assertTrue(all(call.args[0].geometry_only for call in start.call_args_list))

    def test_reference_job_cannot_apply_its_late_result_to_a_different_blob(self):
        with patch("housemaker.merged_generation_workspace._ReferenceGenerationThread", _ReferenceThread):
            self.workspace.generate_reference_button.click()
            runtime = self.workspace._reference_generation_runtime
            self.assertIsNotNone(runtime)
            self.workspace.next_inpaint_button.click()
            self.assertTrue(runtime.thread.interrupted)
            runtime.thread.complete(_reference(99))
        self.assertIsNone(self.objects.get_temporary_object_reference())
        self.assertEqual(self.workspace.video_view.get_selected_blob_index(), 1)

    def test_completed_reference_generation_is_retained_only_for_selected_blob(self):
        with patch("housemaker.merged_generation_workspace._ReferenceGenerationThread", _ReferenceThread):
            self.workspace.generate_reference_button.click()
            runtime = self.workspace._reference_generation_runtime
            self.assertIsNotNone(runtime)
            runtime.thread.complete(_reference(99))
        self.workspace.next_inpaint_button.click()
        self.assertIsNone(self.objects.get_temporary_object_reference())
        self.workspace.previous_inpaint_button.click()
        np.testing.assert_array_equal(self.objects.get_temporary_object_reference(), _reference(99))


# ### Standalone test runner ###
if __name__ == "__main__":
    unittest.main()
