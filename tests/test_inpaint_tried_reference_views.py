"""Tried reference view history belongs to exact, surviving inpaint blobs."""

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


def _stroke(x: float) -> MaskStroke:
    return MaskStroke(MASK_MODE_PAINT, 0.05, (MaskPoint(x, 0.5),))


class _VideoSource:
    def __init__(self) -> None:
        self.metadata = VideoMetadata("tried-reference-views.mp4", 4, 30.0, 200, 100)

    def get_frame(self, frame_index: int) -> np.ndarray:
        return np.full((100, 200, 3), 40 + 40 * frame_index, np.uint8)

    def close(self) -> None:
        pass


# ### Per-blob tried view state tests ###
class InpaintTriedReferenceViewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = GenerationWorkspace()
        self.source = _VideoSource()
        self.workspace._video_source = self.source
        self.workspace._data = GenerationData(video_metadata=self.source.metadata)
        self.workspace.show_frame(0)
        self.workspace.video_view.set_strokes([_stroke(0.2), _stroke(0.8)])
        self.workspace._handle_video_strokes_changed(self.workspace.video_view.get_strokes())
        self.workspace._data.set_frame_strokes(2, [_stroke(0.3)])
        self.keys = self.workspace._current_inpaint_keys()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        APP.processEvents()

    def test_each_blob_keeps_independent_nonpersisted_history(self) -> None:
        before = self.workspace._data.to_dict()
        self.assertEqual(self.workspace.get_inpaint_tried_reference_views(self.keys[0]), frozenset())
        self.assertTrue(self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "front", "Front view"))
        self.assertTrue(self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "top", "Top view"))
        self.assertTrue(self.workspace.mark_inpaint_reference_view_tried(self.keys[2], "side", "Side view"))
        self.workspace.navigate_inpaints(2)
        self.assertEqual(
            self.workspace.get_inpaint_tried_reference_views(self.keys[0]),
            frozenset((("front", "Front view"), ("top", "Top view"))),
        )
        self.assertEqual(self.workspace.get_inpaint_tried_reference_views(self.keys[1]), frozenset())
        self.assertEqual(
            self.workspace.get_inpaint_tried_reference_views(self.keys[2]),
            frozenset((("side", "Side view"),)),
        )
        # Navigation saves current frame selection; history itself is not serialized.
        before["current_frame_index"] = self.workspace._data.current_frame_index
        self.assertEqual(self.workspace._data.to_dict(), before)

    def test_duplicate_attempts_are_idempotent_and_returned_history_is_immutable(self) -> None:
        for _ in range(2):
            self.assertTrue(self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "front", "Front view"))
        history = self.workspace.get_inpaint_tried_reference_views(self.keys[0])
        self.assertIsInstance(history, frozenset)
        self.assertEqual(len(history), 1)

    def test_preset_edits_do_not_mark_the_changed_prompt_as_tried(self) -> None:
        self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "front", "Old front view")
        history = self.workspace.get_inpaint_tried_reference_views(self.keys[0])
        self.assertIn(("front", "Old front view"), history)
        self.assertNotIn(("front", "Improved front view"), history)
        self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "front", "Improved front view")
        self.assertEqual(len(self.workspace.get_inpaint_tried_reference_views(self.keys[0])), 2)

    def test_reads_and_marks_do_not_seek_decode_emit_or_store_strokes(self) -> None:
        changed = Mock()
        self.workspace.inpaint_selection_changed.connect(changed)
        before_frame = self.workspace.video_view.get_frame_bgr()
        before_key = self.workspace.current_inpaint_reference_key()
        before_data = self.workspace._data.to_dict()
        with patch.object(self.workspace, "show_frame", side_effect=AssertionError("No seek")), patch.object(
            self.source, "get_frame", side_effect=AssertionError("No video decode"),
        ), patch.object(
            self.workspace._data, "set_frame_strokes", side_effect=AssertionError("No data write"),
        ):
            self.assertTrue(self.workspace.mark_inpaint_reference_view_tried(self.keys[2], "front", "Front view"))
            self.assertEqual(
                self.workspace.get_inpaint_tried_reference_views(self.keys[2]),
                frozenset((("front", "Front view"),)),
            )
        changed.assert_not_called()
        self.assertEqual(self.workspace._data.to_dict(), before_data)
        self.assertEqual(self.workspace.current_inpaint_reference_key(), before_key)
        np.testing.assert_array_equal(self.workspace.video_view.get_frame_bgr(), before_frame)

    def test_repeated_reads_reuse_the_frame_key_cache(self) -> None:
        self.workspace._inpaint_key_cache.clear()
        with patch.object(
            self.workspace, "_frame_inpaint_masks", wraps=self.workspace._frame_inpaint_masks,
        ) as masks:
            self.workspace.mark_inpaint_reference_view_tried(self.keys[2], "front", "Front view")
            for _ in range(5):
                self.assertEqual(
                    self.workspace.get_inpaint_tried_reference_views(self.keys[2]),
                    frozenset((("front", "Front view"),)),
                )
        masks.assert_called_once_with(2)

    def test_deleted_or_changed_masks_reject_results_and_prune_visible_state(self) -> None:
        self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "front", "Front view")
        self.workspace.mark_inpaint_reference_view_tried(self.keys[2], "top", "Top view")
        self.workspace.video_view.clear_mask()
        self.workspace._data.set_frame_strokes(2, [_stroke(0.6)])
        for key in (self.keys[0], self.keys[2], (3, "missing")):
            self.assertEqual(self.workspace.get_inpaint_tried_reference_views(key), frozenset())
            self.assertFalse(self.workspace.mark_inpaint_reference_view_tried(key, "front", "Front view"))
        self.assertNotIn(self.keys[0], self.workspace._inpaint_tried_reference_views)

    def test_invalid_preset_ids_and_prompts_do_not_change_history(self) -> None:
        self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "  front  ", " Front view ")
        for argument in ("preset", "prompt"):
            for value in (None, False, 1, [], {}):
                args = (value, "Front view") if argument == "preset" else ("front", value)
                with self.subTest(argument=argument, value=value), self.assertRaises(TypeError):
                    self.workspace.mark_inpaint_reference_view_tried(self.keys[0], *args)
            for value in ("", "  ", "\n"):
                args = (value, "Front view") if argument == "preset" else ("front", value)
                with self.subTest(argument=argument, value=value), self.assertRaises(ValueError):
                    self.workspace.mark_inpaint_reference_view_tried(self.keys[0], *args)
        self.assertEqual(
            self.workspace.get_inpaint_tried_reference_views(self.keys[0]),
            frozenset((("front", "Front view"),)),
        )

    def test_project_and_video_replacement_clear_tried_history(self) -> None:
        self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "front", "Front view")
        self.workspace.set_data(GenerationData())
        self.assertFalse(self.workspace._inpaint_tried_reference_views)
        self.workspace._inpaint_tried_reference_views[self.keys[0]] = {("front", "Front view")}
        with patch("housemaker.generation_workspace.probe_video", return_value=self.source.metadata), patch(
            "housemaker.generation_workspace.VideoFrameSource", return_value=_VideoSource(),
        ):
            self.workspace.load_video(self.source.metadata.path)
        self.assertFalse(self.workspace._inpaint_tried_reference_views)
        self.assertEqual(self.workspace.get_inpaint_tried_reference_views(self.keys[0]), frozenset())

    def test_component_mode_does_not_retarget_a_surviving_queued_blob(self) -> None:
        for attribute in ("_door_slot_editing_target", "_architectural_trim_editing_target"):
            with self.subTest(attribute=attribute):
                setattr(self.workspace, attribute, object())
                try:
                    self.assertTrue(self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "front", "Front view"))
                    self.assertEqual(
                        self.workspace.get_inpaint_tried_reference_views(self.keys[0]),
                        frozenset((("front", "Front view"),)),
                    )
                finally:
                    setattr(self.workspace, attribute, None)


# ### Standalone runner ###
if __name__ == "__main__":
    unittest.main()
