"""Owned, selection-independent inputs and results for queued reference jobs."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.video_source import VideoMetadata

# ### Local-only fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _stroke(x: float, y: float = 0.5) -> MaskStroke:
    return MaskStroke(MASK_MODE_PAINT, 0.05, (MaskPoint(x, y),))


def _reference(value: int) -> np.ndarray:
    image = np.full((24, 32, 4), value, np.uint8)
    image[:, :, 3] = 255
    return image


class _VideoSource:
    def __init__(self) -> None:
        self.metadata = VideoMetadata("reference-inputs.mp4", 4, 30.0, 200, 100)
        self.reads: list[int] = []

    def get_frame(self, frame_index: int) -> np.ndarray:
        self.reads.append(frame_index)
        return np.full((100, 200, 3), 40 + 40 * frame_index, np.uint8)

    def close(self) -> None:
        pass


# ### Queue snapshot and result routing tests ###
class InpaintReferenceInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = GenerationWorkspace()
        self.source = _VideoSource()
        self.workspace._video_source = self.source
        self.workspace._data = GenerationData(video_metadata=self.source.metadata)
        self.workspace.show_frame(0)
        self.workspace.video_view.set_strokes([_stroke(0.2), _stroke(0.8)])
        self.workspace._handle_video_strokes_changed(self.workspace.video_view.get_strokes())
        self.workspace._data.set_frame_strokes(2, [_stroke(0.3), _stroke(0.7)])

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        APP.processEvents()

    def test_snapshot_keeps_frame_selection_and_reads_each_other_frame_once(self) -> None:
        self.workspace.video_view.select_blob(1)
        before_frame = self.workspace.video_view.get_frame_bgr()
        before_key = self.workspace.current_inpaint_reference_key()
        self.source.reads.clear()
        with patch.object(self.workspace, "show_frame", side_effect=AssertionError("No navigation")):
            inputs = self.workspace.snapshot_inpaint_reference_inputs()
        self.assertEqual([key[0] for key, _ in inputs], [0, 0, 2, 2])
        self.assertEqual(self.source.reads, [2])
        self.assertEqual(self.workspace.current_inpaint_reference_key(), before_key)
        self.assertEqual(self.workspace._data.current_frame_index, 0)
        np.testing.assert_array_equal(self.workspace.video_view.get_frame_bgr(), before_frame)
        for (_key, reference), expected in zip(inputs, (40, 40, 120, 120)):
            opaque = reference[:, :, 3] > 0
            self.assertTrue(np.any(opaque))
            self.assertTrue(np.all(reference[opaque, :3] == expected))

    def test_snapshot_prefers_each_blobs_own_reference(self) -> None:
        keys = self.workspace._current_inpaint_keys()
        self.workspace.set_inpaint_reference(keys[0], _reference(11))
        self.workspace.set_inpaint_reference(keys[3], _reference(22))
        inputs = self.workspace.snapshot_inpaint_reference_inputs()
        np.testing.assert_array_equal(inputs[0][1], _reference(11))
        np.testing.assert_array_equal(inputs[3][1], _reference(22))
        self.assertNotEqual(inputs[1][1].shape, _reference(11).shape)

    def test_snapshot_and_single_inputs_are_defensive_copies(self) -> None:
        key = self.workspace.current_inpaint_reference_key()
        self.workspace.set_inpaint_reference(key, _reference(51))
        snapshot = self.workspace.snapshot_inpaint_reference_inputs()
        single = self.workspace.get_inpaint_reference_input(key)
        snapshot[0][1][:] = 0
        single[:] = 0
        np.testing.assert_array_equal(self.workspace.get_inpaint_reference_input(key), _reference(51))

    def test_snapshot_stays_owned_after_references_and_masks_change(self) -> None:
        snapshot = self.workspace.snapshot_inpaint_reference_inputs()
        saved = [reference.copy() for _key, reference in snapshot]
        self.workspace.set_inpaint_reference(snapshot[0][0], _reference(99))
        self.workspace.video_view.clear_mask()
        for (_key, reference), expected in zip(snapshot, saved):
            np.testing.assert_array_equal(reference, expected)

    def test_offscreen_result_does_not_replace_the_current_reference_or_signal(self) -> None:
        keys = self.workspace._current_inpaint_keys()
        self.workspace.set_temporary_object_reference(_reference(33))
        changed = Mock()
        self.workspace.inpaint_selection_changed.connect(changed)
        image = _reference(77)
        self.assertTrue(self.workspace.set_inpaint_reference(keys[2], image))
        image[:] = 0
        changed.assert_not_called()
        self.assertEqual(self.workspace._data.current_frame_index, 0)
        np.testing.assert_array_equal(self.workspace.get_temporary_object_reference(), _reference(33))
        np.testing.assert_array_equal(self.workspace.get_inpaint_reference_input(keys[2]), _reference(77))
        self.workspace.navigate_inpaints(2)
        np.testing.assert_array_equal(self.workspace.get_temporary_object_reference(), _reference(77))

    def test_other_blob_on_same_frame_does_not_change_the_selection(self) -> None:
        keys = self.workspace._current_inpaint_keys()
        self.assertTrue(self.workspace.set_inpaint_reference(keys[1], _reference(77)))
        self.assertEqual(self.workspace.current_inpaint_reference_key(), keys[0])
        self.assertIsNone(self.workspace.get_temporary_object_reference())
        self.workspace.navigate_inpaints(1)
        np.testing.assert_array_equal(self.workspace.get_temporary_object_reference(), _reference(77))

    def test_current_result_updates_preview_before_selection_notification(self) -> None:
        key = self.workspace.current_inpaint_reference_key()
        observed = []
        self.workspace.inpaint_selection_changed.connect(lambda: observed.append((
            self.workspace.get_temporary_object_reference(),
            self.workspace.video_view.has_reference_preview(),
        )))
        self.assertTrue(self.workspace.set_inpaint_reference(key, _reference(88)))
        self.assertEqual(len(observed), 1)
        np.testing.assert_array_equal(observed[0][0], _reference(88))
        self.assertTrue(observed[0][1])

    def test_reference_update_invalidates_only_its_fingerprint(self) -> None:
        keys = self.workspace._current_inpaint_keys()
        for key in keys:
            self.workspace.set_inpaint_reference(key, _reference(21))
        before = self.workspace._build_current_generation_mask_signature()
        other_signature = self.workspace._inpaint_reference_signatures[keys[0]]
        self.workspace.set_inpaint_reference(keys[2], _reference(42))
        self.assertNotIn(keys[2], self.workspace._inpaint_reference_signatures)
        self.assertIs(self.workspace._inpaint_reference_signatures[keys[0]], other_signature)
        after = self.workspace._build_current_generation_mask_signature()
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[2], after[2])

    def test_deleted_and_modified_masks_reject_their_old_keys(self) -> None:
        keys = self.workspace._current_inpaint_keys()
        self.workspace.video_view.clear_mask()
        self.workspace._data.set_frame_strokes(2, [_stroke(0.4)])
        for key in (keys[0], keys[2], keys[3]):
            self.assertIsNone(self.workspace.get_inpaint_reference_input(key))
            self.assertFalse(self.workspace.set_inpaint_reference(key, _reference(99)))

    def test_no_selection_still_snapshots_other_frames(self) -> None:
        self.workspace.show_frame(1)
        self.assertFalse(self.workspace.video_view.has_selection())
        self.assertEqual(len(self.workspace.snapshot_inpaint_reference_inputs()), 4)
        self.assertEqual(self.workspace._data.current_frame_index, 1)

    def test_component_mode_blocks_new_snapshots_but_allows_queued_result_storage(self) -> None:
        key = self.workspace.current_inpaint_reference_key()
        self.workspace.set_inpaint_reference(key, _reference(55))
        for attribute in ("_door_slot_editing_target", "_architectural_trim_editing_target"):
            with self.subTest(attribute=attribute):
                setattr(self.workspace, attribute, object())
                try:
                    self.workspace._temporary_object_reference_bgra = _reference(80)
                    self.workspace.video_view.set_reference_preview_bgra(_reference(80))
                    changed = Mock()
                    self.workspace.inpaint_selection_changed.connect(changed)
                    self.assertEqual(self.workspace.snapshot_inpaint_reference_inputs(), ())
                    self.assertIsNotNone(self.workspace.get_inpaint_reference_input(key))
                    self.assertTrue(self.workspace.set_inpaint_reference(key, _reference(99)))
                    np.testing.assert_array_equal(self.workspace.get_temporary_object_reference(), _reference(80))
                    np.testing.assert_array_equal(self.workspace.get_inpaint_reference_input(key), _reference(99))
                    self.assertTrue(self.workspace.video_view.has_reference_preview())
                    changed.assert_not_called()
                    self.workspace.inpaint_selection_changed.disconnect(changed)
                finally:
                    setattr(self.workspace, attribute, None)
        np.testing.assert_array_equal(self.workspace.get_inpaint_reference_input(key), _reference(99))

    def test_invalid_or_transparent_results_do_not_overwrite_references(self) -> None:
        key = self.workspace.current_inpaint_reference_key()
        self.workspace.set_inpaint_reference(key, _reference(55))
        for image in (np.empty((0, 0, 4), np.uint8), np.ones((3, 4, 3), np.uint8), np.zeros((3, 4, 4), np.uint8)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                self.workspace.set_inpaint_reference(key, image)
        np.testing.assert_array_equal(self.workspace.get_inpaint_reference_input(key), _reference(55))


# ### Standalone test runner ###
if __name__ == "__main__":
    unittest.main()
