"""Per-object symmetry decisions stay attached to their exact inpaint masks."""

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
    GeneratedObjectRecord,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.object_symmetry import SYMMETRIC_DIVISION_ORIENTATION_VERTICAL
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.video_source import VideoMetadata

# ### Local-only fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _stroke(x: float) -> MaskStroke:
    return MaskStroke(MASK_MODE_PAINT, 0.05, (MaskPoint(x, 0.5),))


class _VideoSource:
    def __init__(self) -> None:
        self.metadata = VideoMetadata("symmetry-inputs.mp4", 4, 30.0, 200, 100)

    def get_frame(self, frame_index: int) -> np.ndarray:
        return np.full((100, 200, 3), 40 + 40 * frame_index, np.uint8)

    def close(self) -> None:
        pass


# ### Queue state and selection tests ###
class InpaintSymmetryStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = GenerationWorkspace()
        self.source = _VideoSource()
        self.workspace._video_source = self.source
        self.workspace._data = GenerationData(video_metadata=self.source.metadata)
        self.workspace.set_runtime_settings(GenerationServiceSettings(meshy_api_key="test"))
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

    def test_current_result_changes_checked_state_not_widget_availability(self) -> None:
        self.assertIsNone(self.workspace.get_inpaint_symmetric_division(self.keys[0]))
        self.assertTrue(self.workspace.set_inpaint_symmetric_division(self.keys[0], True))
        self.assertTrue(self.workspace.symmetric_division_checkbox.isChecked())
        self.assertTrue(self.workspace.symmetric_division_checkbox.isEnabled())
        self.assertTrue(self.workspace.set_inpaint_symmetric_division(self.keys[0], False))
        self.assertFalse(self.workspace.symmetric_division_checkbox.isChecked())
        self.assertTrue(self.workspace.symmetric_division_checkbox.isEnabled())

    def test_offscreen_results_leave_the_current_view_unchanged(self) -> None:
        changed = Mock()
        self.workspace.inpaint_selection_changed.connect(changed)
        self.assertTrue(self.workspace.set_inpaint_symmetric_division(self.keys[2], True))
        self.assertFalse(self.workspace.symmetric_division_checkbox.isChecked())
        self.assertEqual(self.workspace._data.current_frame_index, 0)
        self.assertEqual(self.workspace.current_inpaint_reference_key(), self.keys[0])
        changed.assert_not_called()
        self.workspace.navigate_inpaints(2)
        self.assertTrue(self.workspace.symmetric_division_checkbox.isChecked())

    def test_navigation_restores_own_inferred_and_manually_overridden_decisions(self) -> None:
        self.workspace.set_inpaint_symmetric_division(self.keys[0], True)
        self.workspace.set_inpaint_symmetric_division(self.keys[1], False)
        self.workspace.navigate_inpaints(1)
        self.assertFalse(self.workspace.symmetric_division_checkbox.isChecked())
        self.workspace.symmetric_division_checkbox.setChecked(True)
        self.assertIs(self.workspace.get_inpaint_symmetric_division(self.keys[1]), True)
        self.workspace.navigate_inpaints(1)
        self.assertFalse(self.workspace.symmetric_division_checkbox.isChecked())
        self.workspace.navigate_inpaints(1)
        self.assertTrue(self.workspace.symmetric_division_checkbox.isChecked())
        self.workspace.navigate_inpaints(1)
        self.assertTrue(self.workspace.symmetric_division_checkbox.isChecked())

    def test_deleted_and_changed_masks_reject_stale_results(self) -> None:
        self.workspace.set_inpaint_symmetric_division(self.keys[0], True)
        self.workspace.video_view.clear_mask()
        self.workspace._data.set_frame_strokes(2, [_stroke(0.6)])
        for key in (self.keys[0], self.keys[2]):
            self.assertIsNone(self.workspace.get_inpaint_symmetric_division(key))
            self.assertFalse(self.workspace.set_inpaint_symmetric_division(key, True))
        self.assertNotIn(self.keys[0], self.workspace._inpaint_symmetric_divisions)
        self.assertFalse(self.workspace.symmetric_division_checkbox.isChecked())

    def test_invalid_values_do_not_mutate_the_decision(self) -> None:
        self.workspace.set_inpaint_symmetric_division(self.keys[0], True)
        for value in (None, 0, 1, "false", np.bool_(False)):
            with self.subTest(value=value), self.assertRaises(TypeError):
                self.workspace.set_inpaint_symmetric_division(self.keys[0], value)
        self.assertIs(self.workspace.get_inpaint_symmetric_division(self.keys[0]), True)

    def test_queued_results_do_not_modify_component_checkboxes(self) -> None:
        for attribute in ("_door_slot_editing_target", "_architectural_trim_editing_target"):
            with self.subTest(attribute=attribute):
                self.workspace.symmetric_division_checkbox.setChecked(False)
                setattr(self.workspace, attribute, object())
                try:
                    self.assertTrue(self.workspace.set_inpaint_symmetric_division(self.keys[0], True))
                    self.assertFalse(self.workspace.symmetric_division_checkbox.isChecked())
                finally:
                    setattr(self.workspace, attribute, None)
                self.workspace._handle_inpaint_selection_changed()
                self.assertTrue(self.workspace.symmetric_division_checkbox.isChecked())

    def test_project_and_video_replacement_clear_nonpersisted_state(self) -> None:
        self.workspace.set_inpaint_symmetric_division(self.keys[0], True)
        self.workspace.set_data(GenerationData())
        self.assertFalse(self.workspace._inpaint_symmetric_divisions)
        self.assertFalse(self.workspace.symmetric_division_checkbox.isChecked())
        self.workspace._inpaint_symmetric_divisions[self.keys[0]] = True
        with patch("housemaker.generation_workspace.probe_video", return_value=self.source.metadata), patch(
            "housemaker.generation_workspace.VideoFrameSource", return_value=_VideoSource(),
        ):
            self.workspace.load_video(self.source.metadata.path)
        self.assertFalse(self.workspace._inpaint_symmetric_divisions)
        self.assertIsNone(self.workspace.get_inpaint_symmetric_division(self.keys[0]))

    # ### Per-object generation snapshots ###
    def test_full_generation_snapshots_each_blob_without_selection_or_image_leaks(self) -> None:
        self.workspace.set_inpaint_symmetric_division(self.keys[0], True)
        self.workspace.set_inpaint_symmetric_division(self.keys[1], False)
        self.workspace.set_inpaint_symmetric_division(self.keys[2], True)
        self.workspace.navigate_inpaints(1)
        requests = self.workspace._build_generation_requests()
        self.assertEqual([request.symmetric_division_enabled for request in requests], [True, False, True])
        self.assertEqual([request.frame_index for request in requests], [0, 0, 2])
        self.assertEqual(self.workspace.current_inpaint_reference_key(), self.keys[1])
        self.assertFalse(self.workspace.symmetric_division_checkbox.isChecked())
        self.workspace.set_inpaint_symmetric_division(self.keys[0], False)
        self.assertTrue(requests[0].symmetric_division_enabled)

    def test_geometry_only_does_not_accidentally_enable_symmetry(self) -> None:
        for key in self.keys:
            self.workspace.set_inpaint_symmetric_division(key, True)
        requests = self.workspace._build_generation_requests(geometry_only=True)
        self.assertEqual([request.symmetric_division_enabled for request in requests], [False] * 3)

    def test_texture_batch_uses_own_decisions_instead_of_the_visible_checkbox(self) -> None:
        self.workspace.set_inpaint_symmetric_division(self.keys[0], True)
        self.workspace.set_inpaint_symmetric_division(self.keys[1], False)
        self.workspace.set_inpaint_symmetric_division(self.keys[2], True)
        self.workspace.navigate_inpaints(1)
        with patch.object(self.workspace, "_start_generation", side_effect=["a", "b", "c"]):
            self.workspace.generate_geometry()
        self.workspace._data.generated_objects = [
            GeneratedObjectRecord(
                object_id=object_id, frame_index=frame_index, object_name=object_id,
                pipeline={}, provider_task_id=object_id, asset_path=f"{object_id}.glb",
            )
            for object_id, frame_index in (("a", 0), ("b", 0), ("c", 2))
        ]
        with patch.object(self.workspace, "_build_texture_regeneration_request", side_effect=[Mock(), Mock(), Mock()]) as build, patch.object(
            self.workspace, "_start_texture_regeneration", return_value=True,
        ):
            self.assertTrue(self.workspace.generate_selected_object_texture())
        self.assertEqual(
            [call.kwargs["symmetric_division_override"] for call in build.call_args_list],
            [True, False, True],
        )

    def test_texture_preflight_supports_new_symmetry_for_normal_objects_only_when_requested(self) -> None:
        self.workspace._data.generated_objects = [GeneratedObjectRecord(
            object_id="object", frame_index=0, object_name="Object", pipeline={},
            provider_task_id="object", asset_path="object.glb",
        )]
        reference = np.full((24, 24, 4), 255, np.uint8)
        with patch.object(self.workspace, "_can_regenerate_object_texture", return_value=True), patch.object(
            self.workspace, "_object_has_complete_texture_uvs", return_value=False,
        ), patch(
            "housemaker.generation_workspace._build_generation_asset_revision",
            return_value=("object.glb", 1, 2, 3),
        ):
            for override in (None, False, True):
                with self.subTest(override=override):
                    request = self.workspace._build_texture_regeneration_request(
                        object_id="object", reference_image_bgra=reference,
                        symmetric_division_override=override,
                    )
                    self.assertIsNotNone(request, self.workspace.status_label.text())
                    self.assertEqual(
                        request.new_symmetric_division_orientation,
                        SYMMETRIC_DIVISION_ORIENTATION_VERTICAL if override else None,
                    )


# ### Standalone runner ###
if __name__ == "__main__":
    unittest.main()
