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
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QWidget

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_JOB_KIND,
    CeilingHeightEstimate,
    CeilingHeightEstimationCancelled,
)
from housemaker.generation_jobs import JOB_STATUS_CANCELLING, GenerationJobManager
from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_workspace import (
    GENERATION_JOB_KIND_MODEL,
    GenerationWorkspace,
)
from housemaker.merged_generation_workspace import (
    CEILING_HEIGHT_NOT_ESTIMATED_TEXT,
    OBJECT_WORKFLOW_OUTLINE_PADDING,
    MergedGenerationWorkspace,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_texture_state import SurfaceTextureData
from housemaker.surface_texture_workspace import (
    SURFACE_TEXTURE_JOB_KIND,
    SurfaceTextureGenerationWorkspace,
)
from housemaker.video_source import VideoMetadata

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Fixtures ###
def _stroke(x_position: float) -> MaskStroke:
    return MaskStroke(
        MASK_MODE_PAINT,
        0.05,
        (MaskPoint(x_position, 0.5),),
    )


def _write_test_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("The test video writer could not be opened.")
    try:
        for frame_index in range(frame_count):
            writer.write(
                np.full(
                    (48, 64, 3),
                    30 + frame_index * 40,
                    dtype=np.uint8,
                )
            )
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


# ### Merged workspace tests ###
class MergedGenerationWorkspaceTests(unittest.TestCase):
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
        self.workspace = MergedGenerationWorkspace(
            self.surfaces,
            self.objects,
            self.job_manager,
        )

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.surfaces.shutdown()
        self.objects.shutdown()
        self.workspace.close()
        _qt_application.processEvents()
        self.temporary_directory.cleanup()

    def test_shared_controls_have_one_authoritative_widget(self) -> None:
        self.assertIsNotNone(self.surfaces._shared_controls)
        self.assertIs(self.objects.video_view, self.surfaces.video_view)
        self.assertIs(
            self.objects.pbr_map_checkboxes,
            self.surfaces.pbr_map_checkboxes,
        )
        self.assertIs(self.objects.ai_prompt_edit, self.surfaces.ai_prompt_edit)
        self.assertIs(
            self.workspace.views_splitter.widget(1),
            self.objects.object_3d_page,
        )
        self.assertFalse(hasattr(self.surfaces, "surface_3d_page"))
        self.assertFalse(hasattr(self.workspace, "preview_panel"))
        self.assertFalse(hasattr(self.objects, "generated_objects_list"))
        self.assertFalse(hasattr(self.objects.object_3d_panel, "object_list"))
        self.assertFalse(hasattr(self.objects, "texture_view"))
        self.assertFalse(hasattr(self.objects, "undo_object_change_button"))
        self.assertFalse(hasattr(self.objects, "symmetric_division_orientation_combo"))
        self.assertFalse(hasattr(self.objects, "frame_label"))
        self.assertFalse(hasattr(self.objects, "job_name_edit"))
        self.assertFalse(hasattr(self.surfaces, "job_name_edit"))
        self.assertFalse(hasattr(self.workspace, "generate_mask"))
        self.assertFalse(hasattr(self.objects, "glass_double_sided_checkbox"))
        self.assertFalse(hasattr(self.objects, "delete_generated_object_button"))
        self.assertFalse(
            hasattr(self.objects.object_3d_panel, "delete_object_button")
        )
        self.assertFalse(
            hasattr(self.objects.object_3d_panel, "face_selection_help_label")
        )
        self.assertFalse(
            hasattr(self.objects.object_3d_panel, "face_selection_count_label")
        )
        object_view_layout = self.objects.object_3d_panel.layout()
        self.assertIsNotNone(object_view_layout)
        assert object_view_layout is not None
        self.assertEqual(object_view_layout.count(), 1)
        self.assertIs(
            object_view_layout.itemAt(0).widget(),
            self.objects.result_view,
        )
        self.assertFalse(hasattr(self.surfaces, "texture_view"))
        self.assertFalse(hasattr(self.surfaces, "delete_texture_button"))

        primary_actions = self.workspace.findChild(
            QWidget,
            "merged_generation_primary_object_actions",
        )
        self.assertIsNotNone(primary_actions)
        assert primary_actions is not None
        self.assertEqual(primary_actions.layout().spacing(), 0)

    def test_current_video_frame_api_tracks_shared_video_view(self) -> None:
        changed = QSignalSpy(self.workspace.current_video_frame_changed)
        frame_bgr = np.zeros((4, 6, 3), dtype=np.uint8)
        frame_bgr[:, :] = (23, 91, 207)

        self.workspace.video_view.set_frame(frame_bgr)

        self.assertEqual(changed.count(), 1)
        returned_frame = self.workspace.get_current_video_frame_bgr()
        self.assertIsNotNone(returned_frame)
        assert returned_frame is not None
        np.testing.assert_array_equal(returned_frame, frame_bgr)
        returned_frame[:] = 0
        np.testing.assert_array_equal(
            self.workspace.get_current_video_frame_bgr(),
            frame_bgr,
        )

        self.workspace.video_view.clear_frame()

        self.assertEqual(changed.count(), 2)
        self.assertIsNone(self.workspace.get_current_video_frame_bgr())

    def test_ceiling_height_control_analyzes_and_displays_current_frame(
        self,
    ) -> None:
        captured_frames: list[np.ndarray] = []
        captured_keys: list[str] = []

        def estimator(
            frame_bgr: np.ndarray,
            *,
            api_key: str,
            cancellation_check=None,
        ) -> CeilingHeightEstimate:
            captured_frames.append(frame_bgr.copy())
            captured_keys.append(api_key)
            frame_bgr[:] = 0
            return CeilingHeightEstimate(
                status="estimated",
                estimate_m=2.6,
                minimum_m=2.4,
                maximum_m=2.8,
                confidence="medium",
                basis="Door proportions and visible room boundaries agree.",
            )

        self.workspace._ceiling_height_estimator = estimator
        self.objects.set_runtime_settings(
            GenerationServiceSettings(openai_api_key="sk-test")
        )
        frame_bgr = np.full((8, 12, 3), 137, dtype=np.uint8)

        self.assertFalse(self.workspace.infer_ceiling_height_button.isEnabled())
        self.workspace.video_view.set_frame(frame_bgr)
        self.assertTrue(self.workspace.infer_ceiling_height_button.isEnabled())

        completed = QSignalSpy(
            self.workspace.ceiling_height_estimate_changed
        )
        self.workspace.infer_ceiling_height_button.click()

        self.assertFalse(self.workspace.infer_ceiling_height_button.isEnabled())
        self.assertEqual(
            self.workspace.infer_ceiling_height_button.text(),
            "Inferring...",
        )
        self.assertTrue(_wait_until(lambda: completed.count() == 1))
        self.assertEqual(captured_keys, ["sk-test"])
        np.testing.assert_array_equal(captured_frames[0], frame_bgr)
        np.testing.assert_array_equal(
            self.workspace.get_current_video_frame_bgr(),
            frame_bgr,
        )
        self.assertIn(
            "Estimated ceiling height: 2.6 m",
            self.workspace.ceiling_height_result_label.text(),
        )
        self.assertTrue(self.workspace.infer_ceiling_height_button.isEnabled())
        job = self.job_manager.jobs()[-1]
        self.assertEqual(job.kind, CEILING_HEIGHT_JOB_KIND)
        self.assertEqual(job.status, "completed")

    def test_frame_change_cancels_and_clears_a_pending_height_estimate(
        self,
    ) -> None:
        started = threading.Event()

        def blocking_estimator(
            _frame_bgr: np.ndarray,
            *,
            api_key: str,
            cancellation_check=None,
        ) -> CeilingHeightEstimate:
            self.assertEqual(api_key, "sk-test")
            started.set()
            while cancellation_check is None or not cancellation_check():
                time.sleep(0.005)
            raise CeilingHeightEstimationCancelled("Cancelled")

        self.workspace._ceiling_height_estimator = blocking_estimator
        self.objects.set_runtime_settings(
            GenerationServiceSettings(openai_api_key="sk-test")
        )
        self.workspace.video_view.set_frame(
            np.full((8, 12, 3), 40, dtype=np.uint8)
        )
        self.workspace.infer_ceiling_height_button.click()
        self.assertTrue(started.wait(1.0))
        self.assertTrue(self.workspace.cancel_button.isEnabled())

        self.workspace.video_view.set_frame(
            np.full((8, 12, 3), 210, dtype=np.uint8)
        )

        self.assertTrue(
            _wait_until(
                lambda: self.workspace._ceiling_height_runtime is None
            )
        )
        self.assertEqual(
            self.workspace.ceiling_height_result_label.text(),
            CEILING_HEIGHT_NOT_ESTIMATED_TEXT,
        )
        self.assertEqual(self.job_manager.jobs()[-1].status, "cancelled")
        self.assertTrue(self.workspace.infer_ceiling_height_button.isEnabled())

    def test_default_window_splits_views_evenly_and_uses_one_shared_column(
        self,
    ) -> None:
        self.workspace.resize(1_600, 900)
        self.workspace.show()
        _qt_application.processEvents()

        video_width = self.workspace.views_splitter.widget(0).width()
        object_width = self.workspace.views_splitter.widget(1).width()
        combined_width = video_width + object_width
        self.assertLessEqual(
            abs(video_width - object_width),
            max(20, round(combined_width * 0.05)),
        )
        self.assertGreater(object_width, 0)
        self.assertFalse(
            self.objects.object_3d_panel.details_panel.isVisibleTo(
                self.objects.object_3d_panel
            )
        )
        self.assertIsNotNone(
            self.workspace.findChild(
                QWidget,
                "merged_generation_shared_primary_column",
            )
        )
        self.assertIsNone(
            self.workspace.findChild(
                QWidget,
                "merged_generation_shared_secondary_column",
            )
        )
        self.assertFalse(
            self.surfaces.status_label.isVisibleTo(self.workspace)
        )
        self.assertFalse(
            self.objects.status_label.isVisibleTo(self.workspace)
        )
        self.assertEqual(
            self.workspace.ceiling_height_result_label.textFormat(),
            Qt.TextFormat.PlainText,
        )
        self.assertGreaterEqual(
            self.workspace.controls_scroll.viewport().height(),
            self.workspace.controls_row.height(),
        )

    def test_object_controls_house_projection_allocation_in_second_column(
        self,
    ) -> None:
        object_fields = self.workspace.findChild(
            QWidget,
            "merged_generation_object_fields",
        )
        primary_column = self.workspace.findChild(
            QWidget,
            "merged_generation_object_primary_column",
        )
        projection_controls = (
            self.objects.object_3d_panel.projection_camera_controls
        )
        self.assertIsNotNone(object_fields)
        self.assertIsNotNone(primary_column)
        assert object_fields is not None
        assert primary_column is not None

        self.assertTrue(object_fields.isAncestorOf(projection_controls))
        self.assertFalse(
            self.objects.object_3d_panel.details_panel.isAncestorOf(
                projection_controls
            )
        )
        fields_layout = object_fields.layout()
        self.assertEqual(
            fields_layout.getItemPosition(fields_layout.indexOf(primary_column)),
            (0, 0, 1, 1),
        )
        self.assertEqual(
            fields_layout.getItemPosition(
                fields_layout.indexOf(projection_controls)
            ),
            (0, 1, 1, 1),
        )
        self.assertEqual(fields_layout.columnStretch(0), 1)
        self.assertEqual(fields_layout.columnStretch(1), 1)

        controls_layout = self.workspace.controls_row.layout()
        self.assertGreaterEqual(
            controls_layout.stretch(2),
            controls_layout.stretch(0),
        )

    def test_object_controls_are_boxed_in_editing_and_generation_sections(
        self,
    ) -> None:
        primary_actions = self.workspace.findChild(
            QWidget,
            "merged_generation_primary_object_actions",
        )
        primary_column = self.workspace.findChild(
            QWidget,
            "merged_generation_object_primary_column",
        )
        generation_settings = self.workspace.findChild(
            QWidget,
            "merged_generation_object_generation_settings",
        )
        generation_actions = self.workspace.findChild(
            QWidget,
            "merged_generation_object_generation_actions",
        )
        face_actions = self.workspace.findChild(
            QWidget,
            "merged_generation_object_face_actions",
        )
        editing_section = self.workspace.findChild(
            QWidget,
            "merged_generation_object_editing_section",
        )
        creation_section = self.workspace.findChild(
            QWidget,
            "merged_generation_object_creation_section",
        )
        self.assertIsNotNone(primary_actions)
        self.assertIsNotNone(primary_column)
        self.assertIsNotNone(generation_settings)
        self.assertIsNotNone(generation_actions)
        self.assertIsNotNone(face_actions)
        self.assertIsNotNone(editing_section)
        self.assertIsNotNone(creation_section)
        assert primary_actions is not None
        assert primary_column is not None
        assert generation_settings is not None
        assert generation_actions is not None
        assert face_actions is not None
        assert editing_section is not None
        assert creation_section is not None

        actions_layout = primary_actions.layout()
        self.assertIs(
            actions_layout.itemAt(0).widget(),
            self.objects.generate_geometry_button,
        )
        self.assertIs(
            actions_layout.itemAt(1).widget(),
            self.objects.generate_texture_button,
        )
        settings_layout = generation_settings.layout()
        self.assertIs(
            settings_layout.itemAt(0).widget(),
            self.objects.symmetric_division_checkbox,
        )
        self.assertIs(
            settings_layout.itemAt(2).widget(),
            self.objects.meshy_target_polycount_control,
        )
        generation_actions_layout = generation_actions.layout()
        self.assertIs(
            generation_actions_layout.itemAt(0).widget(),
            self.objects.generate_button,
        )
        self.assertIs(
            generation_actions_layout.itemAt(1).widget(),
            self.objects.place_button,
        )
        face_actions_layout = face_actions.layout()
        self.assertIs(
            face_actions_layout.itemAt(0).widget(),
            self.objects.delete_selected_faces_button,
        )
        self.assertIs(
            face_actions_layout.itemAt(1).widget(),
            self.objects.convert_faces_to_glass_button,
        )

        self.assertTrue(
            editing_section.isAncestorOf(self.objects.textures_checkbox)
        )
        self.assertTrue(
            editing_section.isAncestorOf(self.objects.wireframe_checkbox)
        )
        self.assertTrue(
            editing_section.isAncestorOf(
                self.objects.delete_selected_faces_button
            )
        )
        self.assertTrue(
            editing_section.isAncestorOf(
                self.objects.convert_faces_to_glass_button
            )
        )
        for generation_control in (
            self.objects.symmetric_division_checkbox,
            self.objects.meshy_target_polycount_control,
            self.objects.generate_geometry_button,
            self.objects.generate_texture_button,
            self.objects.generate_button,
            self.objects.place_button,
        ):
            self.assertTrue(creation_section.isAncestorOf(generation_control))

        primary_layout = primary_column.layout()
        editing_index = primary_layout.indexOf(editing_section)
        creation_index = primary_layout.indexOf(creation_section)
        statistics_index = primary_layout.indexOf(
            self.objects.model_statistics_label
        )
        self.assertLess(editing_index, creation_index)
        self.assertLess(creation_index, statistics_index)

        creation_layout = creation_section.layout()
        self.assertIs(creation_layout.itemAt(0).widget(), generation_settings)
        self.assertIs(creation_layout.itemAt(1).widget(), primary_actions)
        self.assertIs(creation_layout.itemAt(2).widget(), generation_actions)

    def test_material_controls_are_boxed_in_requested_order(self) -> None:
        material_section = self.workspace.findChild(
            QWidget,
            "merged_generation_shared_material_section",
        )
        self.assertIsNotNone(material_section)
        assert material_section is not None

        material_layout = material_section.layout()
        self.assertIs(
            material_layout.itemAt(0).widget(),
            self.objects.pbr_map_control,
        )
        prompt_wrapper = material_layout.itemAt(1).widget()
        self.assertIsNotNone(prompt_wrapper)
        assert prompt_wrapper is not None
        self.assertTrue(prompt_wrapper.isAncestorOf(self.objects.ai_prompt_edit))
        self.assertIs(
            material_layout.itemAt(2).widget(),
            self.workspace.cancel_button,
        )

    def test_object_outline_has_five_pixel_inner_padding(self) -> None:
        self.assertEqual(OBJECT_WORKFLOW_OUTLINE_PADDING, 5)

    def test_seek_stores_each_shared_frame_without_cross_frame_overwrite(self) -> None:
        video_path = Path(self.temporary_directory.name) / "source.avi"
        _write_test_video(video_path)
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))

        first_stroke = _stroke(0.25)
        second_stroke = _stroke(0.75)
        self.workspace.video_view.set_strokes([first_stroke])
        self.workspace.video_view.strokes_changed.emit([first_stroke])
        self.workspace.seekbar.setValue(1)
        self.workspace.video_view.set_strokes([second_stroke])
        self.workspace.video_view.strokes_changed.emit([second_stroke])
        self.workspace.seekbar.setValue(0)

        self.assertEqual(self.objects.get_data().frame_strokes[0], [first_stroke])
        self.assertEqual(self.surfaces.get_data().frame_strokes[0], [first_stroke])
        self.assertEqual(self.objects.get_data().frame_strokes[1], [second_stroke])
        self.assertEqual(self.surfaces.get_data().frame_strokes[1], [second_stroke])
        self.assertEqual(self.workspace.video_view.get_strokes(), [first_stroke])

    def test_clear_mask_keymapping_uses_shared_button_and_can_be_disabled(self) -> None:
        video_path = Path(self.temporary_directory.name) / "source.avi"
        _write_test_video(video_path)
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))
        self.workspace.video_view.set_strokes([_stroke(0.5)])
        self.workspace.sync_shared_controls()

        self.assertTrue(self.workspace.clear_mask_button.isEnabled())
        self.assertEqual(
            self.workspace.clear_mask_shortcut.context(),
            Qt.ShortcutContext.WidgetWithChildrenShortcut,
        )
        self.workspace.set_clear_mask_hotkey("Alt+C")
        self.assertEqual(
            self.workspace.clear_mask_shortcut.key().toString(
                QKeySequence.SequenceFormat.PortableText
            ),
            "Alt+C",
        )
        self.workspace.clear_mask_shortcut.activated.emit()
        self.assertFalse(self.workspace.video_view.has_selection())

        self.workspace.video_view.set_strokes([_stroke(0.5)])
        self.workspace.sync_shared_controls()
        self.workspace.set_clear_mask_hotkey("")
        self.assertFalse(self.workspace.clear_mask_shortcut.isEnabled())
        self.assertTrue(self.workspace.video_view.has_selection())

    def test_cancel_targets_newest_relevant_job_only(self) -> None:
        cancelled: list[str] = []
        first = self.job_manager.create_job(
            kind=GENERATION_JOB_KIND_MODEL,
            requested_name="Object",
            default_name="Object",
            stage="Running",
            cancel_callback=lambda: cancelled.append("object") or True,
        )
        self.job_manager.create_job(
            kind="Atlas ambient occlusion",
            requested_name="AO",
            default_name="AO",
            stage="Running",
            cancel_callback=lambda: cancelled.append("ao") or True,
        )
        second = self.job_manager.create_job(
            kind=SURFACE_TEXTURE_JOB_KIND,
            requested_name="Surface",
            default_name="Surface",
            stage="Running",
            cancel_callback=lambda: cancelled.append("surface") or True,
        )
        self.objects._sync_controls()

        self.assertTrue(self.workspace.cancel_button.isEnabled())

        self.assertTrue(self.workspace.cancel_latest_generation_job())

        self.assertEqual(cancelled, ["surface"])
        self.assertEqual(
            self.job_manager.get_job(second.job_id).status,
            JOB_STATUS_CANCELLING,
        )
        self.assertEqual(
            self.job_manager.get_job(first.job_id).status,
            "running",
        )

    def test_project_migration_merges_legacy_masks_for_the_same_video(self) -> None:
        metadata = VideoMetadata(
            path="reference.avi",
            frame_count=4,
            fps=24.0,
            width=64,
            height=48,
        )
        object_data = GenerationData(
            video_metadata=metadata,
            current_frame_index=1,
            frame_strokes={1: [_stroke(0.25)]},
        )
        surface_data = SurfaceTextureData(
            video_metadata=metadata,
            current_frame_index=2,
            frame_strokes={
                1: [_stroke(0.5)],
                2: [_stroke(0.75)],
            },
        )

        merged_object, merged_surface = (
            MergedGenerationWorkspace.merge_project_video_state(
                object_data,
                surface_data,
            )
        )

        self.assertEqual(merged_object.current_frame_index, 2)
        self.assertEqual(
            merged_object.frame_strokes,
            {
                1: [_stroke(0.5), _stroke(0.25)],
                2: [_stroke(0.75)],
            },
        )
        self.assertEqual(merged_surface.frame_strokes, merged_object.frame_strokes)
        self.assertIsNot(merged_object.frame_strokes, merged_surface.frame_strokes)


if __name__ == "__main__":
    unittest.main()
