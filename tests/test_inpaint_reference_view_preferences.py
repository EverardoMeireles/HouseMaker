"""Manual reference views stay attached to their exact, surviving inpaint blobs."""

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
from housemaker.reference_generation import MAX_REFERENCE_PROMPT_CHARACTERS
from housemaker.video_source import VideoMetadata

# ### Local-only fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _stroke(x: float) -> MaskStroke:
    return MaskStroke(MASK_MODE_PAINT, 0.05, (MaskPoint(x, 0.5),))


class _VideoSource:
    def __init__(self) -> None:
        self.metadata = VideoMetadata("reference-view-preferences.mp4", 4, 30.0, 200, 100)

    def get_frame(self, frame_index: int) -> np.ndarray:
        return np.full((100, 200, 3), 40 + 40 * frame_index, np.uint8)

    def close(self) -> None:
        pass


# ### Per-blob reference view preference tests ###
class InpaintReferenceViewPreferencesTests(unittest.TestCase):
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

    def test_blob_preferences_are_independent_including_other_frames(self) -> None:
        self.assertIsNone(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]))
        self.assertTrue(self.workspace.set_inpaint_reference_view_preferences(
            self.keys[0], "front", "Unused custom prompt",
        ))
        self.assertTrue(self.workspace.set_inpaint_reference_view_preferences(
            self.keys[2], "top", "Another custom prompt",
        ))
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]),
                         ("front", "Unused custom prompt"))
        self.assertIsNone(self.workspace.get_inpaint_reference_view_preferences(self.keys[1]))
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[2]),
                         ("top", "Another custom prompt"))
        self.workspace.navigate_inpaints(2)
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]),
                         ("front", "Unused custom prompt"))

    def test_write_prompt_mode_preserves_custom_text_and_allows_empty_text(self) -> None:
        for prompt in ("", "  Custom view\nwith whitespace  "):
            with self.subTest(prompt=prompt):
                self.assertTrue(self.workspace.set_inpaint_reference_view_preferences(
                    self.keys[0], "", prompt,
                ))
                self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]),
                                 ("", prompt))
        self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "  front  ")
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]),
                         ("front", ""))

    def test_saved_choice_is_replaced_only_for_its_own_blob(self) -> None:
        for key in self.keys:
            self.workspace.set_inpaint_reference_view_preferences(key, "front", "Custom front")
        self.workspace.set_inpaint_reference_view_preferences(self.keys[1], "side", "Custom side")
        self.assertEqual(
            [self.workspace.get_inpaint_reference_view_preferences(key) for key in self.keys],
            [("front", "Custom front"), ("side", "Custom side"), ("front", "Custom front")],
        )

    def test_reads_and_writes_do_not_seek_decode_emit_or_modify_project_data(self) -> None:
        changed = Mock()
        self.workspace.inpaint_selection_changed.connect(changed)
        before_data = self.workspace._data.to_dict()
        before_key = self.workspace.current_inpaint_reference_key()
        before_frame = self.workspace.video_view.get_frame_bgr()
        with patch.object(self.workspace, "show_frame", side_effect=AssertionError("No seek")), patch.object(
            self.source, "get_frame", side_effect=AssertionError("No decode"),
        ), patch.object(
            self.workspace._data, "set_frame_strokes", side_effect=AssertionError("No data write"),
        ):
            self.assertTrue(self.workspace.set_inpaint_reference_view_preferences(self.keys[2], "top"))
            self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[2]),
                             ("top", ""))
        changed.assert_not_called()
        self.assertEqual(self.workspace._data.to_dict(), before_data)
        self.assertEqual(self.workspace.current_inpaint_reference_key(), before_key)
        np.testing.assert_array_equal(self.workspace.video_view.get_frame_bgr(), before_frame)

    def test_repeated_reads_reuse_frame_key_cache(self) -> None:
        self.workspace._inpaint_key_cache.clear()
        with patch.object(self.workspace, "_frame_inpaint_masks",
                          wraps=self.workspace._frame_inpaint_masks) as masks:
            self.workspace.set_inpaint_reference_view_preferences(self.keys[2], "top")
            for _ in range(5):
                self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[2]),
                                 ("top", ""))
        masks.assert_called_once_with(2)

    def test_deleted_and_changed_masks_reject_preferences(self) -> None:
        self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "front")
        self.workspace.set_inpaint_reference_view_preferences(self.keys[2], "top")
        self.workspace.video_view.clear_mask()
        self.workspace._data.set_frame_strokes(2, [_stroke(0.6)])
        for key in (self.keys[0], self.keys[2], (3, "missing")):
            self.assertIsNone(self.workspace.get_inpaint_reference_view_preferences(key))
            self.assertFalse(self.workspace.set_inpaint_reference_view_preferences(key, "side"))
        self.assertNotIn(self.keys[0], self.workspace._inpaint_reference_view_preferences)

    def test_invalid_values_do_not_replace_saved_preference(self) -> None:
        self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "front", "Saved")
        for argument in ("preset", "prompt"):
            for value in (None, False, 1, [], {}):
                args = (value, "Custom") if argument == "preset" else ("front", value)
                with self.subTest(argument=argument, value=value), self.assertRaises(TypeError):
                    self.workspace.set_inpaint_reference_view_preferences(self.keys[0], *args)
        with self.assertRaises(ValueError):
            self.workspace.set_inpaint_reference_view_preferences(
                self.keys[0], "", "x" * (MAX_REFERENCE_PROMPT_CHARACTERS + 1),
            )
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]),
                         ("front", "Saved"))
        boundary_prompt = "x" * MAX_REFERENCE_PROMPT_CHARACTERS
        self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "", boundary_prompt)
        self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]),
                         ("", boundary_prompt))

    def test_project_and_video_replacement_clear_session_preferences(self) -> None:
        self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "front")
        self.workspace.set_data(GenerationData())
        self.assertFalse(self.workspace._inpaint_reference_view_preferences)
        self.workspace._inpaint_reference_view_preferences[self.keys[0]] = ("front", "Custom")
        with patch("housemaker.generation_workspace.probe_video", return_value=self.source.metadata), patch(
            "housemaker.generation_workspace.VideoFrameSource", return_value=_VideoSource(),
        ):
            self.workspace.load_video(self.source.metadata.path)
        self.assertFalse(self.workspace._inpaint_reference_view_preferences)
        self.assertIsNone(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]))

    def test_component_mode_does_not_retarget_an_explicit_blob_key(self) -> None:
        for attribute in ("_door_slot_editing_target", "_architectural_trim_editing_target"):
            with self.subTest(attribute=attribute):
                setattr(self.workspace, attribute, object())
                try:
                    self.assertTrue(self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "front"))
                    self.assertEqual(self.workspace.get_inpaint_reference_view_preferences(self.keys[0]),
                                     ("front", ""))
                finally:
                    setattr(self.workspace, attribute, None)

    def test_manual_choice_does_not_modify_tried_view_history(self) -> None:
        self.workspace.mark_inpaint_reference_view_tried(self.keys[0], "front", "Front view")
        self.workspace.set_inpaint_reference_view_preferences(self.keys[0], "side")
        self.assertEqual(self.workspace.get_inpaint_tried_reference_views(self.keys[0]),
                         frozenset((("front", "Front view"),)))


# ### Standalone runner ###
if __name__ == "__main__":
    unittest.main()
