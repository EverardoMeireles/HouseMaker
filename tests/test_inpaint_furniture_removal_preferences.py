"""Furniture content removal stays local to each surviving inpaint blob."""

# ### Environment and imports ###
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from test_inpaint_reference_view_preferences import APP, _stroke, _VideoSource

from housemaker.generation_state import GenerationData
from housemaker.generation_workspace import GenerationWorkspace


# ### Per-blob furniture object removal tests ###
class InpaintFurnitureRemovalPreferencesTests(unittest.TestCase):
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

    def test_preferences_are_independent_with_true_false_and_unset_values(self) -> None:
        self.assertIsNone(self.workspace.get_inpaint_remove_objects_from_furniture(self.keys[0]))
        self.assertTrue(self.workspace.set_inpaint_remove_objects_from_furniture(self.keys[0], True))
        self.assertTrue(self.workspace.set_inpaint_remove_objects_from_furniture(self.keys[2], False))
        self.assertEqual(
            [self.workspace.get_inpaint_remove_objects_from_furniture(key) for key in self.keys],
            [True, None, False],
        )
        self.workspace.navigate_inpaints(2)
        self.assertTrue(self.workspace.get_inpaint_remove_objects_from_furniture(self.keys[0]))
        self.assertFalse(self.workspace.get_inpaint_remove_objects_from_furniture(self.keys[2]))
        self.workspace.set_inpaint_remove_objects_from_furniture(self.keys[0], False)
        self.assertEqual(
            [self.workspace.get_inpaint_remove_objects_from_furniture(key) for key in self.keys],
            [False, None, False],
        )

    def test_reads_and_writes_are_silent_without_seeking_or_project_mutation(self) -> None:
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
            for key in (self.keys[0], self.keys[2]):
                self.assertTrue(self.workspace.set_inpaint_remove_objects_from_furniture(key, True))
                self.assertTrue(self.workspace.get_inpaint_remove_objects_from_furniture(key))
        changed.assert_not_called()
        self.assertEqual(self.workspace._data.to_dict(), before_data)
        self.assertEqual(self.workspace.current_inpaint_reference_key(), before_key)
        np.testing.assert_array_equal(self.workspace.video_view.get_frame_bgr(), before_frame)

    def test_repeated_reads_reuse_frame_key_cache(self) -> None:
        self.workspace._inpaint_key_cache.clear()
        with patch.object(self.workspace, "_frame_inpaint_masks",
                          wraps=self.workspace._frame_inpaint_masks) as masks:
            self.workspace.set_inpaint_remove_objects_from_furniture(self.keys[2], True)
            for _ in range(5):
                self.assertTrue(self.workspace.get_inpaint_remove_objects_from_furniture(self.keys[2]))
        masks.assert_called_once_with(2)

    def test_deleted_changed_and_missing_masks_reject_preferences(self) -> None:
        for key in self.keys:
            self.workspace.set_inpaint_remove_objects_from_furniture(key, True)
        self.workspace.video_view.clear_mask()
        self.workspace._data.set_frame_strokes(2, [_stroke(0.6)])
        for key in (self.keys[0], self.keys[2], (3, "missing")):
            self.assertIsNone(self.workspace.get_inpaint_remove_objects_from_furniture(key))
            self.assertFalse(self.workspace.set_inpaint_remove_objects_from_furniture(key, False))
        self.assertNotIn(self.keys[0], self.workspace._inpaint_remove_objects_from_furniture)
        self.assertTrue(self.workspace.get_inpaint_remove_objects_from_furniture(self.keys[1]))

    def test_non_boolean_values_do_not_replace_saved_preference(self) -> None:
        self.workspace.set_inpaint_remove_objects_from_furniture(self.keys[0], True)
        for value in (None, 0, 1, "false", [], {}, np.bool_(True)):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.workspace.set_inpaint_remove_objects_from_furniture(self.keys[0], value)
        self.assertTrue(self.workspace.get_inpaint_remove_objects_from_furniture(self.keys[0]))

    def test_project_and_video_replacement_clear_session_preferences(self) -> None:
        self.workspace.set_inpaint_remove_objects_from_furniture(self.keys[0], True)
        self.workspace.set_data(GenerationData())
        self.assertFalse(self.workspace._inpaint_remove_objects_from_furniture)
        self.workspace._inpaint_remove_objects_from_furniture[self.keys[0]] = True
        with patch("housemaker.generation_workspace.probe_video", return_value=self.source.metadata), patch(
            "housemaker.generation_workspace.VideoFrameSource", return_value=_VideoSource(),
        ):
            self.workspace.load_video(self.source.metadata.path)
        self.assertFalse(self.workspace._inpaint_remove_objects_from_furniture)
        self.assertIsNone(self.workspace.get_inpaint_remove_objects_from_furniture(self.keys[0]))

    def test_component_mode_does_not_retarget_explicit_blob_keys(self) -> None:
        for attribute in ("_door_slot_editing_target", "_architectural_trim_editing_target"):
            with self.subTest(attribute=attribute):
                setattr(self.workspace, attribute, object())
                try:
                    self.assertTrue(self.workspace.set_inpaint_remove_objects_from_furniture(self.keys[0], True))
                    self.assertTrue(self.workspace.get_inpaint_remove_objects_from_furniture(self.keys[0]))
                finally:
                    setattr(self.workspace, attribute, None)


# ### Standalone runner ###
if __name__ == "__main__":
    unittest.main()
