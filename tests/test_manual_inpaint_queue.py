# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ### Imports ###
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtWidgets import QApplication

from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GeneratedObjectRecord,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_views import rasterize_mask_strokes
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.video_source import VideoMetadata

# ### Fixtures ###
_application = QApplication.instance() or QApplication([])


def _stroke(x: float, y: float) -> MaskStroke:
    return MaskStroke(MASK_MODE_PAINT, 0.035, (MaskPoint(x, y),))


def _reference(value: int) -> np.ndarray:
    image = np.full((28, 36, 4), value, dtype=np.uint8)
    image[:, :, 3] = 255
    return image


class _VideoSource:
    def __init__(self) -> None:
        self.metadata = VideoMetadata("manual-test.mp4", 6, 30.0, 200, 100)
        self.read_frames: list[int] = []

    def get_frame(self, index: int) -> np.ndarray:
        self.read_frames.append(index)
        return np.full((100, 200, 3), 30 + 20 * index, dtype=np.uint8)

    def close(self) -> None:
        pass


# ### Cross-frame queue tests ###
class ManualInpaintQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = GenerationWorkspace()
        self.source = _VideoSource()
        self.workspace._video_source = self.source
        self.workspace._data = GenerationData(video_metadata=self.source.metadata)
        self.workspace.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test"))
        self.workspace.show_frame(0)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        _application.processEvents()

    def _paint(self, *points: tuple[float, float]) -> None:
        self.workspace.video_view.set_strokes([_stroke(*point) for point in points])
        self.workspace._handle_video_strokes_changed(self.workspace.video_view.get_strokes())

    def _three_objects(self) -> None:
        self._paint((0.2, 0.25), (0.8, 0.75))
        self.workspace._data.set_frame_strokes(3, [_stroke(0.5, 0.5)])

    def test_queue_orders_all_frames_and_independent_blobs(self) -> None:
        self._three_objects()
        self.assertEqual(self.workspace.inpaint_queue(), ((0, 0), (0, 1), (3, 0)))
        self.assertEqual(self.workspace.current_inpaint_position(), (0, 3))

    def test_navigation_wraps_and_preserves_other_frames(self) -> None:
        self._three_objects()
        self.assertTrue(self.workspace.navigate_inpaints(-1))
        self.assertEqual(self.workspace.current_inpaint_position(), (2, 3))
        self.assertEqual(self.workspace._data.current_frame_index, 3)
        self.assertTrue(self.workspace.navigate_inpaints(1))
        self.assertEqual(self.workspace.current_inpaint_position(), (0, 3))
        self.assertEqual(len(self.workspace._data.strokes_for_frame(0)), 2)

    def test_navigation_from_empty_frame_selects_first_or_last(self) -> None:
        self._three_objects()
        self.workspace.show_frame(2)
        self.assertEqual(self.workspace.current_inpaint_position(), (None, 3))
        self.assertTrue(self.workspace.navigate_inpaints(1))
        self.assertEqual(self.workspace.current_inpaint_position(), (0, 3))
        self.workspace.show_frame(2)
        self.assertTrue(self.workspace.navigate_inpaints(-1))
        self.assertEqual(self.workspace.current_inpaint_position(), (2, 3))

    def test_no_masks_has_no_navigation(self) -> None:
        self.assertEqual(self.workspace.inpaint_queue(), ())
        self.assertEqual(self.workspace.current_inpaint_position(), (None, 0))
        self.assertFalse(self.workspace.navigate_inpaints(1))

    def test_removing_last_object_disables_generation_despite_erase_history(self) -> None:
        self._paint((0.3, 0.3))
        self.assertTrue(self.workspace.generate_button.isEnabled())
        self.workspace.video_view.clear_mask()
        self.assertTrue(self.workspace._data.frame_strokes)
        self.assertEqual(self.workspace.inpaint_queue(), ())
        self.assertFalse(self.workspace.generate_button.isEnabled())

    def test_generation_snapshots_all_frames_without_display_navigation(self) -> None:
        self._three_objects()
        self.workspace.show_frame(2)
        before_frame = self.workspace.video_view.get_frame_bgr()
        requests = self.workspace._build_generation_requests()
        self.assertEqual([request.frame_index for request in requests], [0, 0, 3])
        self.assertEqual(self.workspace._data.current_frame_index, 2)
        np.testing.assert_array_equal(self.workspace.video_view.get_frame_bgr(), before_frame)
        for request, expected in zip(requests, [30, 30, 90]):
            opaque = request.selected_object_bgra[:, :, 3] > 0
            self.assertTrue(np.all(request.selected_object_bgra[opaque, :3] == expected))
        self.assertIs(requests[0].settings, requests[2].settings)
        self.assertTrue(self.workspace.generate_button.isEnabled())

    def test_geometry_and_full_generation_launch_each_queued_object(self) -> None:
        self._three_objects()
        self.workspace.show_frame(2)
        for action, geometry_only in (("generate", False), ("generate_geometry", True)):
            with self.subTest(action=action), patch.object(
                self.workspace, "_start_generation", side_effect=["a", "b", "c"],
            ) as start:
                getattr(self.workspace, action)()
                self.assertEqual(start.call_count, 3)
                self.assertEqual([call.args[0].frame_index for call in start.call_args_list], [0, 0, 3])
                self.assertTrue(all(call.args[0].geometry_only is geometry_only for call in start.call_args_list))

    def test_private_references_restore_only_for_their_own_blob(self) -> None:
        self._three_objects()
        self.workspace.set_temporary_object_reference(_reference(210))
        self.assertTrue(self.workspace.navigate_inpaints(1))
        self.assertFalse(self.workspace.has_temporary_object_reference())
        self.workspace.set_temporary_object_reference(_reference(160))
        self.workspace.navigate_inpaints(1)
        self.assertFalse(self.workspace.has_temporary_object_reference())
        self.workspace.navigate_inpaints(1)
        np.testing.assert_array_equal(self.workspace.get_temporary_object_reference(), _reference(210))
        requests = self.workspace._build_generation_requests()
        np.testing.assert_array_equal(requests[0].selected_object_bgra, _reference(210))
        np.testing.assert_array_equal(requests[1].selected_object_bgra, _reference(160))
        self.assertNotEqual(requests[2].selected_object_bgra.shape, _reference(210).shape)

    def test_seeking_away_and_back_preserves_private_reference(self) -> None:
        self._paint((0.3, 0.3))
        self.workspace.set_temporary_object_reference(_reference(180))
        self.workspace.show_frame(2)
        self.assertFalse(self.workspace.has_temporary_object_reference())
        self.workspace.show_frame(0)
        np.testing.assert_array_equal(self.workspace.get_temporary_object_reference(), _reference(180))
        self.assertTrue(self.workspace.video_view.has_reference_preview())

    def test_merged_direct_frame_changes_restore_reference_before_signal(self) -> None:
        self._paint((0.3, 0.3))
        self.workspace.set_temporary_object_reference(_reference(180))
        observed = []
        self.workspace.inpaint_selection_changed.connect(
            lambda: observed.append(self.workspace.get_temporary_object_reference())
        )
        for index in (2, 0):
            self.workspace._data.current_frame_index = index
            self.workspace._displayed_frame_index = index
            self.workspace.video_view.set_frame(self.source.get_frame(index), self.workspace._data.strokes_for_frame(index))
        self.assertIsNone(observed[0])
        np.testing.assert_array_equal(observed[-1], _reference(180))

    def test_removing_selected_blob_keeps_other_frames_and_clears_its_reference(self) -> None:
        self._three_objects()
        self.workspace.set_temporary_object_reference(_reference(190))
        removed_key = self.workspace.current_inpaint_reference_key()
        self.workspace.video_view.clear_mask()
        self.assertEqual(self.workspace.inpaint_queue(), ((0, 0), (3, 0)))
        self.assertNotIn(removed_key, self.workspace._inpaint_references)
        self.assertFalse(self.workspace.has_temporary_object_reference())

    def test_signature_covers_all_frames_and_references_not_current_selection(self) -> None:
        self._three_objects()
        signature = self.workspace._build_current_generation_mask_signature()
        self.workspace.show_frame(2)
        self.assertEqual(signature, self.workspace._build_current_generation_mask_signature())
        self.workspace.navigate_inpaints(1)
        self.workspace.set_temporary_object_reference(_reference(220))
        self.assertNotEqual(signature, self.workspace._build_current_generation_mask_signature())

    def test_request_images_are_owned_after_reference_or_mask_changes(self) -> None:
        self._three_objects()
        self.workspace.set_temporary_object_reference(_reference(220))
        requests = self.workspace._build_generation_requests()
        self.workspace.set_temporary_object_reference(_reference(100))
        self.workspace.video_view.clear_mask()
        np.testing.assert_array_equal(requests[0].selected_object_bgra, _reference(220))
        self.assertEqual(len(requests), 3)

    def test_standalone_reference_still_generates_one_object(self) -> None:
        self.workspace.set_temporary_object_reference(_reference(240))
        requests = self.workspace._build_generation_requests()
        self.assertEqual(len(requests), 1)
        np.testing.assert_array_equal(requests[0].selected_object_bgra, _reference(240))

    def test_texture_batch_uses_correct_references_after_navigating(self) -> None:
        self._three_objects()
        self.workspace.set_temporary_object_reference(_reference(200))
        with patch.object(self.workspace, "_start_generation", side_effect=["a", "b", "c"]):
            self.workspace.generate_geometry()
        self.workspace._data.generated_objects = [
            GeneratedObjectRecord(
                object_id=object_id, frame_index=frame_index, object_name=object_id,
                pipeline={}, provider_task_id=object_id, asset_path=f"{object_id}.glb",
            )
            for object_id, frame_index in (("a", 0), ("b", 0), ("c", 3))
        ]
        self.workspace.show_frame(2)
        self.assertTrue(self.workspace.can_generate_inpaint_batch_texture())
        with patch.object(self.workspace, "_build_texture_regeneration_request", side_effect=[Mock(), Mock(), Mock()]) as build, patch.object(
            self.workspace, "_start_texture_regeneration", return_value=True,
        ) as start:
            self.assertTrue(self.workspace.generate_selected_object_texture())
        self.assertEqual(start.call_count, 3)
        calls = build.call_args_list
        self.assertEqual([call.kwargs["object_id"] for call in calls], ["a", "b", "c"])
        self.assertEqual([call.kwargs["reference_frame_index"] for call in calls], [0, 0, 3])
        np.testing.assert_array_equal(calls[0].kwargs["reference_image_bgra"], _reference(200))

    def test_new_project_discards_private_references(self) -> None:
        self._paint((0.3, 0.3))
        self.workspace.set_temporary_object_reference(_reference(100))
        self.workspace.set_data(GenerationData())
        self.assertEqual(self.workspace._inpaint_references, {})
        self.assertFalse(self.workspace.has_temporary_object_reference())

    def test_repeated_queue_queries_reuse_connectivity_counts(self) -> None:
        self._three_objects()
        with patch(
            "housemaker.generation_workspace.rasterize_mask_strokes",
            wraps=rasterize_mask_strokes,
        ) as rasterize:
            expected = self.workspace.inpaint_queue()
            first_count = rasterize.call_count
            for _ in range(8):
                self.assertEqual(self.workspace.inpaint_queue(), expected)
                self.workspace.current_inpaint_position()
            self.assertEqual(rasterize.call_count, first_count)
            self.workspace._data.set_frame_strokes(3, [_stroke(0.7, 0.7)])
            self.workspace.inpaint_queue()
            self.assertEqual(rasterize.call_count, first_count + 1)

    def test_component_reference_never_overwrites_selected_inpaint_reference(self) -> None:
        self._paint((0.3, 0.3))
        self.workspace.set_temporary_object_reference(_reference(200))
        key = self.workspace.current_inpaint_reference_key()
        self.assertTrue(self.workspace.set_door_slot_editing_target(
            door_id="door", slot_id="door_knob", object_id="knob",
            object_name="Handle", slot_display_name="Handle",
        ))
        self.workspace.set_temporary_object_reference(_reference(80))
        self.assertIsNone(self.workspace.current_inpaint_reference_key())
        np.testing.assert_array_equal(self.workspace._inpaint_references[key], _reference(200))
        self.workspace.clear_door_slot_editing_target()
        np.testing.assert_array_equal(self.workspace.get_temporary_object_reference(), _reference(200))
        self.workspace.show_frame(2)
        self.workspace.show_frame(0)
        np.testing.assert_array_equal(self.workspace.get_temporary_object_reference(), _reference(200))

    def test_clearing_component_reference_keeps_inpaint_reference(self) -> None:
        self._paint((0.3, 0.3))
        self.workspace.set_temporary_object_reference(_reference(200))
        self.workspace.set_door_slot_editing_target(
            door_id="door", slot_id="door_knob", object_id="knob",
            object_name="Handle", slot_display_name="Handle",
        )
        self.workspace.set_temporary_object_reference(_reference(80))
        self.workspace.clear_temporary_object_reference()
        self.workspace.clear_door_slot_editing_target()
        np.testing.assert_array_equal(self.workspace.get_temporary_object_reference(), _reference(200))

    def test_bitmap_cache_is_bounded_while_counts_remain_available(self) -> None:
        self.source.metadata = VideoMetadata("manual-test.mp4", 30, 30.0, 200, 100)
        self.workspace._data.video_metadata = self.source.metadata
        for frame_index in range(1, 20):
            self.workspace._data.set_frame_strokes(frame_index, [_stroke(0.5, 0.5)])
        self.assertEqual(len(self.workspace.inpaint_queue()), 19)
        self.assertLessEqual(len(self.workspace._inpaint_component_cache), 4)
        self.assertEqual(len(self.workspace._inpaint_count_cache), 19)


# ### Entry point ###
if __name__ == "__main__":
    unittest.main()
