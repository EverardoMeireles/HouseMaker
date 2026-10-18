# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.ceiling_height_dialog import (
    CEILING_HEIGHT_INTERVAL_SETTING_KEY,
    CeilingHeightAnalysisDialog,
    _level_range_color,
)
from housemaker.ceiling_height_ranges import CeilingHeightVideoRange
from housemaker.video_source import VideoMetadata

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Independent video-reader fixture ###
class FakeVideoFrameSource:
    instances: ClassVar[list[FakeVideoFrameSource]] = []

    def __init__(self, path: str) -> None:
        if "broken" in path:
            raise ValueError("Unreadable video")
        frame_count = 0 if "empty" in path else 350
        self.metadata = VideoMetadata(
            str(Path(path).resolve()),
            frame_count,
            30.0,
            64,
            32,
        )
        self.read_frames: list[int] = []
        self.closed = False
        self.instances.append(self)

    def get_frame(self, frame: int) -> np.ndarray:
        self.read_frames.append(frame)
        return np.full((32, 64, 3), frame % 256, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


# ### Dialog tests ###
class CeilingHeightAnalysisDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.settings = ApplicationSettingsStore(
            Path(self.temporary_directory.name) / "settings.json",
        )
        self.source_patch = patch(
            "housemaker.ceiling_height_dialog.VideoFrameSource",
            FakeVideoFrameSource,
        )
        self.source_patch.start()
        FakeVideoFrameSource.instances.clear()
        self.dialog = CeilingHeightAnalysisDialog(application_settings=self.settings)
        self.dialog.set_level_options(
            (("L1 Underground", 0), ("L2 Ground", 1), ("L3 Story", 2)),
            current_level_index=1,
        )

    def tearDown(self) -> None:
        self.dialog.dispose()
        _qt_application.processEvents()
        self.source_patch.stop()
        self.temporary_directory.cleanup()

    def load_video(self, *, frame: int = 0) -> None:
        self.dialog.set_video_path(
            Path(self.temporary_directory.name) / "tour.mp4",
            current_frame=frame,
        )

    def mark_range(self, start: int, end: int, *, level: int = 1) -> None:
        self.dialog.level_to_mark_combo.setCurrentIndex(
            self.dialog.level_to_mark_combo.findData(level),
        )
        self.dialog.set_current_frame(start)
        self.dialog.mark_begin_button.click()
        self.dialog.set_current_frame(end)
        self.dialog.mark_end_button.click()

    def test_model_and_current_level_fields_are_removed(self) -> None:
        self.assertFalse(hasattr(self.dialog, "model_combo"))
        self.assertFalse(hasattr(self.dialog, "current_level_combo"))
        self.assertEqual(self.dialog.frame_analysis_interval, 15)
        self.assertEqual(self.dialog.selected_level_index, 1)
        self.assertFalse(self.dialog.analysis_button.isEnabled())

    def test_source_opens_independently_and_seek_displays_frames(self) -> None:
        source_changed = QSignalSpy(self.dialog.source_changed)
        self.load_video(frame=24)
        self.assertEqual(len(FakeVideoFrameSource.instances), 1)
        reader = FakeVideoFrameSource.instances[0]
        self.assertEqual(reader.read_frames[-1], 24)
        self.assertEqual(self.dialog.seek_slider.value(), 24)
        self.assertEqual(source_changed.count(), 1)
        self.assertFalse(self.dialog.video_preview_label.pixmap().isNull())

        self.dialog.seek_slider.setValue(99)
        self.assertEqual(reader.read_frames[-1], 99)
        self.assertIn("100 / 350", self.dialog.current_frame_label.text())

    def test_multiple_ranges_per_level_emit_model_interval_and_marks(self) -> None:
        requested = QSignalSpy(self.dialog.analysis_requested)
        self.load_video()
        self.mark_range(0, 60)
        self.mark_range(100, 145)
        self.mark_range(180, 240, level=2)
        expected = (
            CeilingHeightVideoRange(1, 0, 60),
            CeilingHeightVideoRange(1, 100, 145),
            CeilingHeightVideoRange(2, 180, 240),
        )
        self.assertEqual(self.dialog.marked_ranges, expected)
        self.dialog.analysis_button.click()
        self.assertEqual(requested.count(), 1)
        self.assertEqual(list(requested.at(0)), ["gpt-5.6-luna", 15, expected])

    def test_reversed_begin_and_end_are_normalized(self) -> None:
        self.load_video()
        self.mark_range(90, 30)
        self.assertEqual(
            self.dialog.marked_ranges, (CeilingHeightVideoRange(1, 30, 90),)
        )
        self.assertIn("No unfinished", self.dialog.pending_mark_label.text())

    def test_pending_mark_retains_its_original_level(self) -> None:
        self.load_video(frame=10)
        self.dialog.mark_begin_button.click()
        self.assertIn("L2 Ground", self.dialog.pending_mark_label.text())
        self.dialog.level_to_mark_combo.setCurrentIndex(2)
        self.dialog.set_current_frame(40)
        self.dialog.mark_end_button.click()
        self.assertEqual(
            self.dialog.marked_ranges, (CeilingHeightVideoRange(1, 10, 40),)
        )

    def test_cross_level_overlaps_are_rejected_without_losing_previous_marks(
        self,
    ) -> None:
        self.load_video()
        self.mark_range(10, 80)
        self.mark_range(60, 100, level=2)
        self.assertEqual(
            self.dialog.marked_ranges, (CeilingHeightVideoRange(1, 10, 80),)
        )
        self.assertIn("cannot overlap", self.dialog.status_label.text())
        self.assertTrue(self.dialog.mark_end_button.isEnabled())

    def test_programmatic_marks_clamp_to_the_loaded_video(self) -> None:
        self.load_video()
        self.dialog.set_marked_ranges((CeilingHeightVideoRange(1, 300, 600),))
        self.assertEqual(
            self.dialog.marked_ranges, (CeilingHeightVideoRange(1, 300, 349),)
        )

    def test_timeline_endpoints_can_be_dragged_to_extend_and_shrink(self) -> None:
        self.load_video()
        self.mark_range(50, 150)
        self.dialog.show()
        _qt_application.processEvents()
        timeline = self.dialog.range_timeline
        y = round(timeline._range_rect(self.dialog.marked_ranges[0]).center().y())
        start = QPoint(round(timeline._frame_x(50)), y)
        extended = QPoint(round(timeline._frame_x(20)), y)
        QTest.mousePress(timeline, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(timeline, extended)
        QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, pos=extended)
        self.assertAlmostEqual(self.dialog.marked_ranges[0].start_frame, 20, delta=1)

        end = QPoint(round(timeline._frame_x(150)), y)
        shrunk = QPoint(round(timeline._frame_x(100)), y)
        QTest.mousePress(timeline, Qt.MouseButton.LeftButton, pos=end)
        QTest.mouseMove(timeline, shrunk)
        QTest.mouseRelease(timeline, Qt.MouseButton.LeftButton, pos=shrunk)
        self.assertAlmostEqual(self.dialog.marked_ranges[0].end_frame, 100, delta=1)

    def test_selected_range_can_be_removed(self) -> None:
        self.load_video()
        self.mark_range(20, 60)
        self.dialog.remove_range_button.click()
        self.assertEqual(self.dialog.marked_ranges, ())
        self.assertFalse(self.dialog.analysis_button.isEnabled())

    def test_same_source_keeps_marks_new_source_closes_reader_and_clears_marks(
        self,
    ) -> None:
        self.load_video()
        self.mark_range(10, 40)
        old_reader = FakeVideoFrameSource.instances[0]
        self.load_video(frame=20)
        self.assertEqual(len(FakeVideoFrameSource.instances), 1)
        self.assertEqual(len(self.dialog.marked_ranges), 1)

        self.dialog.set_video_path(Path(self.temporary_directory.name) / "second.mp4")
        self.assertTrue(old_reader.closed)
        self.assertEqual(self.dialog.marked_ranges, ())

    def test_failed_source_change_preserves_current_source_and_marks(self) -> None:
        self.load_video()
        self.mark_range(10, 40)
        path = self.dialog.video_path
        with self.assertRaisesRegex(ValueError, "Unreadable"):
            self.dialog.set_video_path("broken.mp4")
        self.assertEqual(self.dialog.video_path, path)
        self.assertEqual(len(self.dialog.marked_ranges), 1)
        self.assertFalse(FakeVideoFrameSource.instances[0].closed)

    def test_empty_source_is_closed_without_reading_or_losing_existing_marks(
        self,
    ) -> None:
        self.load_video()
        self.mark_range(10, 40)
        path = self.dialog.video_path
        with self.assertRaisesRegex(ValueError, "does not contain readable frames"):
            self.dialog.set_video_path("empty.mp4")
        empty_reader = FakeVideoFrameSource.instances[-1]
        self.assertTrue(empty_reader.closed)
        self.assertEqual(empty_reader.read_frames, [])
        self.assertEqual(self.dialog.video_path, path)
        self.assertEqual(
            self.dialog.marked_ranges, (CeilingHeightVideoRange(1, 10, 40),)
        )

    def test_level_colors_are_distinct_beyond_six_and_stable_across_ordering(
        self,
    ) -> None:
        indices = tuple(range(-2, 10))
        colors = {index: _level_range_color(index).name() for index in indices}
        self.assertEqual(len(set(colors.values())), len(indices))
        self.dialog.set_level_options(tuple((f"Level {i}", i) for i in indices))
        self.dialog.set_level_options(
            tuple((f"Level {i}", i) for i in reversed(indices))
        )
        self.assertEqual(
            {index: _level_range_color(index).name() for index in reversed(indices)},
            colors,
        )

    def test_running_disables_source_and_marks_but_allows_cancel(self) -> None:
        self.load_video()
        self.mark_range(0, 60)
        cancelled = QSignalSpy(self.dialog.cancellation_requested)
        self.dialog.set_running(True)
        self.dialog.set_progress(2, 5, current_frame=30)
        self.assertFalse(self.dialog.load_video_button.isEnabled())
        self.assertFalse(self.dialog.mark_begin_button.isEnabled())
        self.assertFalse(self.dialog.mark_end_button.isEnabled())
        self.assertFalse(self.dialog.level_to_mark_combo.isEnabled())
        self.assertTrue(self.dialog.seek_slider.isEnabled())
        self.dialog.analysis_button.click()
        self.assertEqual(cancelled.count(), 1)
        self.assertEqual(self.dialog.progress_bar.maximum(), 5)
        self.assertEqual(self.dialog.progress_bar.value(), 2)
        self.assertEqual(
            self.dialog.status_label.text(),
            "Analysed 2 of 5 frame intervals (video frame 31)",
        )
        self.dialog.set_running(False)
        self.assertTrue(self.dialog.progress_bar.isHidden())

    def test_progress_distinguishes_sample_intervals_from_video_frames(self) -> None:
        self.dialog.set_progress(90, 101, current_frame=1350)
        self.assertEqual(
            self.dialog.status_label.text(),
            "Analysed 90 of 101 frame intervals (video frame 1,351)",
        )
        self.dialog.set_progress(101, 101)
        self.assertEqual(
            self.dialog.status_label.text(),
            "Analysed 101 of 101 frame intervals",
        )

    def test_model_precedes_cost_and_cost_uses_marked_samples_only(self) -> None:
        self.load_video()
        self.dialog.set_video_metadata(350, fps=30, width=1920, height=1080)
        self.mark_range(0, 349)
        all_frames_text = self.dialog.analysis_button.text()
        self.assertRegex(
            all_frames_text, r"\(gpt-5\.6-luna; estimated cost.*\$\d+\.\d{2}.*\)$"
        )
        self.dialog.set_marked_ranges((CeilingHeightVideoRange(1, 0, 45),))
        short_text = self.dialog.analysis_button.text()
        self.assertNotEqual(short_text, all_frames_text)
        self.dialog.set_model("qwen3-vl-8b-local")
        self.assertIn("qwen3-vl-8b-local; Local", self.dialog.analysis_button.text())
        self.assertNotIn("$", self.dialog.analysis_button.text())

    def test_interval_is_persisted_model_is_owned_by_settings(self) -> None:
        self.dialog.set_model("gpt-5.6-terra")
        self.dialog.frame_interval_spinbox.setValue(45)
        restored = CeilingHeightAnalysisDialog(application_settings=self.settings)
        try:
            self.assertEqual(restored.frame_analysis_interval, 45)
            self.assertEqual(restored.selected_model, "gpt-5.6-luna")
            self.assertEqual(self.settings.get(CEILING_HEIGHT_INTERVAL_SETTING_KEY), 45)
        finally:
            restored.dispose()
            _qt_application.processEvents()

    def test_removed_level_marks_are_pruned_and_emitted(self) -> None:
        self.load_video()
        self.mark_range(0, 45)
        self.mark_range(60, 90, level=2)
        changed = QSignalSpy(self.dialog.ranges_changed)
        self.dialog.set_level_options((("L2 Ground", 1),))
        self.assertEqual(
            self.dialog.marked_ranges, (CeilingHeightVideoRange(1, 0, 45),)
        )
        self.assertEqual(changed.count(), 1)

    def test_close_stops_playback_and_dispose_closes_the_reader(self) -> None:
        self.load_video()
        reader = FakeVideoFrameSource.instances[0]
        self.dialog.show()
        self.dialog.play_button.click()
        self.assertTrue(self.dialog._play_timer.isActive())
        self.dialog.close()
        self.assertFalse(self.dialog._play_timer.isActive())
        self.assertFalse(self.dialog.isVisible())
        self.assertFalse(reader.closed)
        self.dialog.dispose()
        self.assertTrue(reader.closed)


# ### Test runner ###
if __name__ == "__main__":
    unittest.main()
