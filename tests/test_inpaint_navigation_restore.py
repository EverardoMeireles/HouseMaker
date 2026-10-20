"""Saved manual inpaints remain navigable beside passive opening previews."""

# ### Environment and imports ###
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from housemaker.generation_jobs import GenerationJobManager
from housemaker.generation_state import GenerationData, MaskPoint, MaskStroke
from housemaker.generation_workspace import GenerationWorkspace
from housemaker.merged_generation_workspace import MergedGenerationWorkspace
from housemaker.surface_texture_state import SurfaceTextureData
from housemaker.surface_texture_workspace import SurfaceTextureGenerationWorkspace
from housemaker.video_source import probe_video

# ### Local fixtures ###
APP = QApplication.instance() or QApplication([])
APP.setQuitOnLastWindowClosed(False)


def _stroke(x: float) -> MaskStroke:
    return MaskStroke("paint", 0.07, (MaskPoint(x, 0.5),))


# ### Restored navigation regressions ###
class InpaintNavigationRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.manager = GenerationJobManager()
        self.objects = GenerationWorkspace(
            asset_directory=root / "objects", job_manager=self.manager,
        )
        self.surfaces = SurfaceTextureGenerationWorkspace(
            asset_directory=root / "surfaces", job_manager=self.manager,
            shared_controls=self.objects.get_shared_controls(),
        )
        self.workspace = MergedGenerationWorkspace(
            self.surfaces, self.objects, self.manager,
        )
        video = root / "saved.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48),
        )
        self.assertTrue(writer.isOpened())
        try:
            for value in (40, 80, 120):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        self.saved = GenerationData(
            video_metadata=probe_video(str(video)), current_frame_index=0,
            frame_strokes={0: [_stroke(0.25), _stroke(0.75)], 2: [_stroke(0.5)]},
        )
        objects, surfaces = self.workspace.merge_project_video_state(
            self.saved, SurfaceTextureData(),
        )
        # Match BlueprintWorkspace's project restoration order.
        self.objects.set_data(objects, defer_file_backed_preview=True)
        self.surfaces.set_data(surfaces)

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.objects.shutdown()
        self.surfaces.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        APP.processEvents()
        self.temporary.cleanup()

    def _preview_opening(self, kind: str = "door", *, active: bool = False) -> None:
        self.assertTrue(self.objects.set_door_slot_editing_target(
            door_id="saved-opening", slot_id="body",
            object_id=f"{kind}:saved-opening:slot:body",
            object_name="Saved opening", slot_display_name="Body",
            asset_kind=kind, generation_ready=active,
            defer_file_backed_preview=True,
        ))

    def test_saved_blobs_are_navigable_immediately_with_passive_opening_preview(self) -> None:
        for kind in ("door", "window"):
            with self.subTest(kind=kind):
                self._preview_opening(kind)
                self.assertTrue(self.workspace.previous_inpaint_button.isEnabled())
                self.assertTrue(self.workspace.next_inpaint_button.isEnabled())
                self.assertEqual(self.workspace.inpaint_position_label.text(), "Inpaint 1 of 3")
                self.assertEqual(self.objects.inpaint_queue(), ((0, 0), (0, 1), (2, 0)))
                self.assertEqual(self.objects._data.frame_strokes, self.saved.frame_strokes)

    def test_next_on_same_frame_leaves_passive_preview_and_selects_second_blob(self) -> None:
        self._preview_opening()
        self.workspace.next_inpaint_button.click()
        self.assertIsNone(self.objects._door_slot_editing_target)
        self.assertEqual(self.objects._displayed_frame_index, 0)
        self.assertEqual(self.workspace.video_view.get_selected_blob_index(), 1)
        self.assertEqual(self.workspace.inpaint_position_label.text(), "Inpaint 2 of 3")
        self.assertIsNotNone(self.objects.current_inpaint_reference_key())

    def test_previous_wraps_to_saved_blob_without_first_seeking_manually(self) -> None:
        self._preview_opening("window")
        self.workspace.previous_inpaint_button.click()
        self.assertIsNone(self.objects._door_slot_editing_target)
        self.assertEqual(self.objects._displayed_frame_index, 2)
        self.assertEqual(self.surfaces._displayed_frame_index, 2)
        self.assertEqual(self.workspace.inpaint_position_label.text(), "Inpaint 3 of 3")

    def test_explicit_opening_generation_keeps_navigation_disabled(self) -> None:
        self._preview_opening(active=True)
        self.assertFalse(self.workspace.previous_inpaint_button.isEnabled())
        self.assertFalse(self.workspace.next_inpaint_button.isEnabled())
        self.assertEqual(self.objects.inpaint_queue(), ((0, 0), (0, 1), (2, 0)))

    def test_passive_preview_does_not_enable_navigation_without_blobs(self) -> None:
        self.saved.frame_strokes = {}
        objects, surfaces = self.workspace.merge_project_video_state(
            self.saved, SurfaceTextureData(),
        )
        self.objects.set_data(objects)
        self.surfaces.set_data(surfaces)
        self._preview_opening()
        self.assertFalse(self.workspace.previous_inpaint_button.isEnabled())
        self.assertFalse(self.workspace.next_inpaint_button.isEnabled())


# ### Standalone test runner ###
if __name__ == "__main__":
    unittest.main()
