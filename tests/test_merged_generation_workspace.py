# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QWidget

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.ceiling_height_estimation import (
    CEILING_HEIGHT_JOB_KIND,
    CeilingHeightEstimate,
    CeilingHeightEstimationCancelled,
    CeilingHeightEstimationError,
)
from housemaker.door_state import DOOR_SLOT_BODY, DOOR_SLOT_KNOB
from housemaker.generation_jobs import JOB_STATUS_CANCELLING, GenerationJobManager
from housemaker.generation_state import (
    MASK_MODE_PAINT,
    GenerationData,
    MaskPoint,
    MaskStroke,
)
from housemaker.generation_workspace import (
    GENERATION_JOB_KIND_MODEL,
    GENERATION_JOB_KIND_TEXTURE,
    GenerationWorkspace,
)
from housemaker.merged_generation_workspace import (
    GENERATION_CONTROLS_PADDING,
    MergedGenerationWorkspace,
    _CeilingHeightInferenceThread,
    _invoke_bound_ceiling_height_estimator,
)
from housemaker.models import LevelData
from housemaker.plan_metric_state import (
    PlanMeasurementEvidence,
    PlanMetricCalibration,
)
from housemaker.qwen_ceiling_height_estimation import (
    LOCAL_QWEN_CEILING_HEIGHT_MODEL,
)
from housemaker.settings_widget import GenerationServiceSettings
from housemaker.surface_geometry import FixedSurface
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
                    min(250, 30 + frame_index * 20),
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


def _plan_metric_calibration(
    label: str,
    value_meters: float,
) -> PlanMetricCalibration:
    """Build distinct, valid plan evidence for routing tests."""

    measurements = tuple(
        PlanMeasurementEvidence(
            label=measurement_label,
            value_meters=measurement_value,
            start_normalized=start,
            end_normalized=end,
            corrected_start_normalized=start,
            corrected_end_normalized=end,
            confidence="high",
        )
        for measurement_label, measurement_value, start, end in (
            (label, value_meters, (0.1, 0.25), (0.9, 0.25)),
            (
                f"{label} secondary",
                value_meters * 0.75,
                (0.2, 0.35),
                (0.8, 0.35),
            ),
        )
    )
    return PlanMetricCalibration(
        estimated_meters_per_pixel=0.02,
        target_meters_per_pixel=0.02,
        normalization_scale=1.0,
        relative_residual=0.01,
        confidence="high",
        source_image_size=(1_000, 800),
        pre_normalization_size=(1_000, 800),
        normalized_image_size=(1_000, 800),
        measurements=measurements,
    )


def _calibrated_ceiling_levels() -> tuple[LevelData, ...]:
    """Return three levels with distinguishable horizontal plan evidence."""

    return (
        LevelData(
            index=1,
            name="Underground 1",
            plan_metric_calibration=_plan_metric_calibration(
                "Underground span",
                3.2,
            ),
        ),
        LevelData(
            index=2,
            name="Ground",
            plan_metric_calibration=_plan_metric_calibration(
                "Ground span",
                6.1,
            ),
        ),
        LevelData(
            index=3,
            name="Story 1",
            plan_metric_calibration=_plan_metric_calibration(
                "Story span",
                3.9,
            ),
        ),
    )


class _ActionJobGenerationWorkspace(GenerationWorkspace):
    """Replace provider work with deterministic jobs for action-button tests."""

    def __init__(self, *args, **kwargs) -> None:
        self.action_job_ids: dict[str, list[str]] = {}
        self.cancelled_actions: list[str] = []
        self.skip_next_action: str | None = None
        super().__init__(*args, **kwargs)

    def _create_action_jobs(
        self,
        action: str,
        kind: str,
        *,
        count: int = 1,
    ) -> list[str]:
        if self.skip_next_action == action:
            self.skip_next_action = None
            return []
        manager = self._job_manager
        assert manager is not None
        job_ids: list[str] = []
        for job_index in range(1, count + 1):
            action_member = f"{action}:{job_index}"
            job = manager.create_job(
                kind=kind,
                requested_name=action_member,
                default_name=action_member,
                stage="Running",
                cancel_callback=(
                    lambda member=action_member: (
                        self.cancelled_actions.append(member) or True
                    )
                ),
            )
            job_ids.append(job.job_id)
        self.action_job_ids[action] = job_ids
        return job_ids

    def generate(self) -> None:
        self._create_action_jobs(
            "generate",
            GENERATION_JOB_KIND_MODEL,
            count=2,
        )

    def generate_geometry(self) -> None:
        self._create_action_jobs(
            "geometry",
            GENERATION_JOB_KIND_MODEL,
        )

    def generate_selected_object_texture(self) -> bool:
        return bool(
            self._create_action_jobs(
                "object_texture",
                GENERATION_JOB_KIND_TEXTURE,
            )
        )


class _ActionJobSurfaceWorkspace(SurfaceTextureGenerationWorkspace):
    """Replace Surface provider work with one deterministic managed job."""

    def __init__(self, *args, **kwargs) -> None:
        self.action_job_ids: list[str] = []
        self.cancelled_actions: list[str] = []
        super().__init__(*args, **kwargs)

    def generate(self) -> None:
        job = self._job_manager.create_job(
            kind=SURFACE_TEXTURE_JOB_KIND,
            requested_name="surface_texture",
            default_name="surface_texture",
            stage="Running",
            cancel_callback=(
                lambda: self.cancelled_actions.append("surface_texture") or True
            ),
        )
        self.action_job_ids = [job.job_id]


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
        self.workspace.set_ceiling_height_levels(
            (
                LevelData(index=1, name="Underground 1"),
                LevelData(index=2, name="Ground"),
                LevelData(index=3, name="Story 1"),
            ),
            current_level_index=2,
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
        self.assertFalse(hasattr(self.workspace, "reference_editing_section"))
        self.assertEqual(
            self.workspace.reference_generation_section.title(),
            "Reference generation",
        )
        self.assertIs(
            self.workspace.findChild(
                QWidget,
                "merged_generation_reference_generation_section",
            ),
            self.workspace.reference_generation_section,
        )
        self.assertIsNone(
            self.workspace.findChild(
                QWidget,
                "merged_generation_object_reference_editing_section",
            )
        )
        self.assertIsNone(
            self.workspace.findChild(QLabel, "merged_generation_shared_title")
        )
        self.assertIsNone(
            self.workspace.findChild(QLabel, "merged_generation_object_title")
        )

        primary_actions = self.workspace.findChild(
            QWidget,
            "merged_generation_primary_object_actions",
        )
        self.assertIsNotNone(primary_actions)
        assert primary_actions is not None
        self.assertEqual(primary_actions.layout().spacing(), 0)

    def test_door_slot_generation_indicator_blinks_beside_heading(self) -> None:
        self.objects.door_slot_generation_status_changed.emit(
            "Door knob/handle"
        )

        self.assertEqual(
            self.workspace.door_slot_generation_indicator.text(),
            "- Generating the door's Door knob/handle",
        )
        self.assertFalse(
            self.workspace.door_slot_generation_indicator.isHidden()
        )
        self.assertTrue(
            self.workspace._door_slot_generation_blink_timer.isActive()
        )

        self.objects.door_slot_generation_status_changed.emit("")

        self.assertTrue(
            self.workspace.door_slot_generation_indicator.isHidden()
        )
        self.assertFalse(
            self.workspace._door_slot_generation_blink_timer.isActive()
        )

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

    def test_ceiling_height_levels_build_metric_context_per_level(self) -> None:
        self.workspace.set_ceiling_height_levels(
            iter(_calibrated_ceiling_levels()),
            current_level_index=2,
        )

        contexts = dict(self.workspace._ceiling_height_level_plan_contexts)

        self.assertEqual(tuple(contexts), (1, 2, 3))
        self.assertIn("M1 - Underground span: 3.2 m", contexts[1])
        self.assertIn("M1 - Ground span: 6.1 m", contexts[2])
        self.assertIn("M1 - Story span: 3.9 m", contexts[3])
        for context in contexts.values():
            self.assertIn("horizontal plan distances only", context)
            self.assertIn("never a ceiling-height measurement", context)

    def test_whole_video_inference_routes_context_after_level_movement(
        self,
    ) -> None:
        levels = _calibrated_ceiling_levels()
        for level in levels:
            level.image_path = str(
                Path(self.temporary_directory.name)
                / f"level-{level.index}.png"
            )
        self.workspace.set_ceiling_height_levels(levels, current_level_index=2)
        movements = (
            ("stable", "high"),
            ("arrived_up", "high"),
            ("stable", "high"),
            ("arrived_down", "medium"),
            ("arrived_down", "medium"),
            ("stable", "high"),
        )
        captured_contexts: list[str] = []

        @contextmanager
        def estimator_provider(**_kwargs):
            def estimator(
                _frame_bgr: np.ndarray,
                *,
                previous_frame_bgr: np.ndarray | None = None,
                plan_context: str = "",
                cancellation_check=None,
            ) -> CeilingHeightEstimate:
                del previous_frame_bgr, cancellation_check
                call_index = len(captured_contexts)
                captured_contexts.append(plan_context)
                movement, confidence = movements[call_index]
                return CeilingHeightEstimate(
                    status="estimated",
                    estimate_m=2.6,
                    minimum_m=2.4,
                    maximum_m=2.8,
                    confidence="medium",
                    basis="Visible room boundaries agree.",
                    level_movement=movement,
                    movement_confidence=confidence,
                )

            yield estimator

        video_path = Path(self.temporary_directory.name) / "context-video.avi"
        _write_test_video(video_path, frame_count=len(movements))
        thread = _CeilingHeightInferenceThread(
            str(video_path),
            tuple(range(len(movements))),
            "sk-test",
            "gpt-5.6-luna",
            None,
            self.workspace._ceiling_height_level_options,
            self.workspace._ceiling_height_level_plan_contexts,
            2,
            level_plan_sources=(
                self.workspace._ceiling_height_level_plan_sources
            ),
        )

        with (
            patch(
                "housemaker.merged_generation_workspace."
                "open_ceiling_height_estimator",
                estimator_provider,
            ),
            patch(
                "housemaker.merged_generation_workspace."
                "build_ceiling_height_plan_image_data_uri",
                return_value="data:image/jpeg;base64,cGxhbg==",
            ),
        ):
            thread.run()

        self.assertIsNone(thread.error_message)
        self.assertIsNotNone(thread.result)
        self.assertEqual(len(captured_contexts), len(movements))
        self.assertIn("Ground span: 6.1 m", captured_contexts[0])
        self.assertIn("Ground span: 6.1 m", captured_contexts[1])
        self.assertIn("Story span: 3.9 m", captured_contexts[2])
        self.assertIn("Story span: 3.9 m", captured_contexts[3])
        self.assertIn("Ground span: 6.1 m", captured_contexts[4])
        self.assertIn("Ground span: 6.1 m", captured_contexts[5])

    def test_metric_context_keeps_legacy_bound_estimator_compatible(
        self,
    ) -> None:
        captured_previous_frames: list[np.ndarray | None] = []

        def legacy_estimator(
            _frame_bgr: np.ndarray,
            *,
            previous_frame_bgr: np.ndarray | None = None,
            cancellation_check=None,
        ) -> CeilingHeightEstimate:
            del cancellation_check
            captured_previous_frames.append(previous_frame_bgr)
            return CeilingHeightEstimate(
                status="estimated",
                estimate_m=2.6,
                minimum_m=2.4,
                maximum_m=2.8,
                confidence="medium",
                basis="Visible room boundaries agree.",
            )

        previous_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        estimate = _invoke_bound_ceiling_height_estimator(
            legacy_estimator,
            np.ones((4, 4, 3), dtype=np.uint8),
            previous_frame=previous_frame,
            plan_context="Metric floor-plan context for Ground: 6.1 m.",
            plan_image_data_uri=None,
            cancellation_check=lambda: False,
        )

        self.assertTrue(estimate.has_estimate)
        self.assertEqual(len(captured_previous_frames), 1)
        self.assertIs(captured_previous_frames[0], previous_frame)

    def test_ceiling_height_control_analyzes_complete_video_with_frame_pairs(
        self,
    ) -> None:
        provider_calls: list[tuple[str, str]] = []
        captured_pairs: list[tuple[float | None, float]] = []
        movements = (
            ("stable", "low", 2.6),
            ("stable", "low", 2.6),
            ("stable", "low", 2.6),
            ("ascending", "medium", 2.8),
            ("ascending", "medium", 2.8),
            ("stable", "low", 3.0),
            ("stable", "low", 3.0),
            ("stable", "low", 3.0),
        )

        @contextmanager
        def estimator_provider(
            *,
            model: str,
            api_key: str,
            estimator_override=None,
            startup_cancellation_check=None,
        ):
            del estimator_override, startup_cancellation_check
            provider_calls.append((model, api_key))

            def estimator(
                frame_bgr: np.ndarray,
                *,
                previous_frame_bgr: np.ndarray | None = None,
                cancellation_check=None,
            ) -> CeilingHeightEstimate:
                del cancellation_check
                pair_index = len(captured_pairs)
                captured_pairs.append(
                    (
                        None
                        if previous_frame_bgr is None
                        else float(previous_frame_bgr.mean()),
                        float(frame_bgr.mean()),
                    )
                )
                movement, movement_confidence, estimate_m = movements[
                    pair_index
                ]
                return CeilingHeightEstimate(
                    status="estimated",
                    estimate_m=estimate_m,
                    minimum_m=estimate_m - 0.2,
                    maximum_m=estimate_m + 0.2,
                    confidence="medium",
                    basis="Visible room boundaries and door scale agree.",
                    level_movement=movement,
                    movement_confidence=movement_confidence,
                )

            yield estimator

        self.objects.set_runtime_settings(
            GenerationServiceSettings(openai_api_key="sk-test")
        )
        video_path = Path(self.temporary_directory.name) / "height-video.avi"
        _write_test_video(video_path, frame_count=14)

        self.assertFalse(self.workspace.infer_ceiling_height_button.isEnabled())
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))
        self.workspace.sync_shared_controls()
        self.assertTrue(self.workspace.infer_ceiling_height_button.isEnabled())

        completed = QSignalSpy(
            self.workspace.ceiling_height_estimate_changed
        )
        self.workspace.infer_ceiling_height_button.click()
        dialog = self.workspace._ceiling_height_dialog
        self.assertIsNotNone(dialog)
        assert dialog is not None
        self.assertFalse(hasattr(dialog, "mark_start_button"))
        self.assertFalse(hasattr(dialog, "mark_end_button"))
        self.assertEqual(dialog.current_level_combo.currentData(), 2)
        self.assertEqual(
            tuple(
                dialog.current_level_combo.itemText(index)
                for index in range(dialog.current_level_combo.count())
            ),
            ("L3 Story 1", "L2 Ground", "L1 Underground 1"),
        )
        dialog.frame_interval_spinbox.setValue(2)
        with patch(
            "housemaker.merged_generation_workspace."
            "open_ceiling_height_estimator",
            estimator_provider,
        ):
            dialog.analysis_button.click()
            self.assertTrue(_wait_until(lambda: completed.count() == 1))

        self.assertEqual(provider_calls, [("gpt-5.6-luna", "sk-test")])
        self.assertEqual(len(captured_pairs), len(movements))
        self.assertIsNone(captured_pairs[0][0])
        for pair_index in range(1, len(captured_pairs)):
            self.assertAlmostEqual(
                captured_pairs[pair_index][0],
                captured_pairs[pair_index - 1][1],
                places=3,
            )
        self.assertIn(
            "L2 Ground: 2.60 m",
            self.workspace.ceiling_height_result_label.text(),
        )
        self.assertIn(
            "L3 Story 1: 3.00 m",
            self.workspace.ceiling_height_result_label.text(),
        )
        result = completed.at(0)[0]
        self.assertEqual(
            result.sampled_frame_indices,
            (0, 2, 4, 6, 8, 10, 12, 13),
        )
        self.assertEqual(result.transition_frame_indices, (6, 8))
        self.assertTrue(self.workspace.infer_ceiling_height_button.isEnabled())
        job = self.job_manager.jobs()[-1]
        self.assertEqual(job.kind, CEILING_HEIGHT_JOB_KIND)
        self.assertEqual(job.status, "completed")

    def test_frame_change_does_not_cancel_pending_video_height_analysis(
        self,
    ) -> None:
        started = threading.Event()
        release = threading.Event()
        call_count = 0

        def blocking_estimator(
            _frame_bgr: np.ndarray,
            *,
            api_key: str,
            model: str,
            cancellation_check=None,
        ) -> CeilingHeightEstimate:
            nonlocal call_count
            self.assertEqual(api_key, "sk-test")
            self.assertEqual(model, "gpt-5.6-luna")
            call_count += 1
            started.set()
            while not release.is_set():
                if cancellation_check is not None and cancellation_check():
                    raise CeilingHeightEstimationCancelled("Cancelled")
                time.sleep(0.005)
            return CeilingHeightEstimate(
                status="estimated",
                estimate_m=2.7,
                minimum_m=2.5,
                maximum_m=2.9,
                confidence="medium",
                basis="Visible room boundaries and door scale agree.",
            )

        self.workspace._ceiling_height_estimator = blocking_estimator
        self.objects.set_runtime_settings(
            GenerationServiceSettings(openai_api_key="sk-test")
        )
        video_path = Path(self.temporary_directory.name) / "height-seek.avi"
        _write_test_video(video_path, frame_count=3)
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))
        self.workspace.sync_shared_controls()
        self.workspace.infer_ceiling_height_button.click()
        dialog = self.workspace._ceiling_height_dialog
        assert dialog is not None
        dialog.frame_interval_spinbox.setValue(2)
        dialog.analysis_button.click()
        self.assertTrue(started.wait(1.0))
        self.workspace.seekbar.setValue(1)
        self.assertIsNotNone(self.workspace._ceiling_height_runtime)
        release.set()
        self.assertTrue(
            _wait_until(
                lambda: self.workspace._ceiling_height_runtime is None
            )
        )
        self.assertEqual(call_count, 2)
        self.assertIn(
            "L2 Ground: 2.70 m",
            self.workspace.ceiling_height_result_label.text(),
        )
        self.assertEqual(self.job_manager.jobs()[-1].status, "completed")
        self.assertTrue(self.workspace.infer_ceiling_height_button.isEnabled())

    def test_isolated_frame_failure_does_not_abort_video_analysis(self) -> None:
        call_count = 0

        def estimator(
            _frame_bgr: np.ndarray,
            *,
            api_key: str,
            model: str,
            cancellation_check=None,
        ) -> CeilingHeightEstimate:
            nonlocal call_count
            del api_key, model, cancellation_check
            call_count += 1
            if call_count == 1:
                raise CeilingHeightEstimationError(
                    "The first sampled frame could not be analysed."
                )
            return CeilingHeightEstimate(
                status="estimated",
                estimate_m=2.7,
                minimum_m=2.5,
                maximum_m=2.9,
                confidence="medium",
                basis="Visible room boundaries and door scale agree.",
            )

        self.workspace._ceiling_height_estimator = estimator
        self.objects.set_runtime_settings(
            GenerationServiceSettings(openai_api_key="sk-test")
        )
        video_path = Path(self.temporary_directory.name) / "height-retry.avi"
        _write_test_video(video_path, frame_count=4)
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))
        self.workspace.sync_shared_controls()

        completed = QSignalSpy(
            self.workspace.ceiling_height_estimate_changed
        )
        self.workspace.infer_ceiling_height_button.click()
        dialog = self.workspace._ceiling_height_dialog
        assert dialog is not None
        dialog.frame_interval_spinbox.setValue(1)
        dialog.analysis_button.click()

        self.assertTrue(_wait_until(lambda: completed.count() == 1))
        self.assertEqual(call_count, 4)
        result = completed.at(0)[0]
        self.assertEqual(result.unassigned_frame_indices, (0,))
        self.assertIn(
            "L2 Ground: 2.70 m",
            self.workspace.ceiling_height_result_label.text(),
        )
        self.assertEqual(self.job_manager.jobs()[-1].status, "completed")

    def test_local_qwen_analysis_does_not_require_an_openai_key(self) -> None:
        captured_keys: list[str] = []
        captured_models: list[str] = []

        def local_estimator(
            _frame_bgr: np.ndarray,
            *,
            api_key: str,
            model: str,
            cancellation_check=None,
        ) -> CeilingHeightEstimate:
            del cancellation_check
            captured_keys.append(api_key)
            captured_models.append(model)
            return CeilingHeightEstimate(
                status="estimated",
                estimate_m=2.7,
                minimum_m=2.5,
                maximum_m=2.9,
                confidence="medium",
                basis="Visible room boundaries and door proportions agree.",
            )

        self.workspace._ceiling_height_estimator = local_estimator
        self.objects.set_runtime_settings(GenerationServiceSettings())
        video_path = Path(self.temporary_directory.name) / "local-height.avi"
        _write_test_video(video_path, frame_count=3)
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))
        self.workspace.sync_shared_controls()

        completed = QSignalSpy(
            self.workspace.ceiling_height_estimate_changed
        )
        self.workspace.infer_ceiling_height_button.click()
        dialog = self.workspace._ceiling_height_dialog
        self.assertIsNotNone(dialog)
        assert dialog is not None
        model_index = dialog.model_combo.findData(
            LOCAL_QWEN_CEILING_HEIGHT_MODEL
        )
        self.assertGreaterEqual(model_index, 0)
        dialog.model_combo.setCurrentIndex(model_index)
        dialog.frame_interval_spinbox.setValue(2)
        dialog.analysis_button.click()

        self.assertTrue(_wait_until(lambda: completed.count() == 1))
        self.assertEqual(captured_keys, ["", ""])
        self.assertEqual(
            captured_models,
            [LOCAL_QWEN_CEILING_HEIGHT_MODEL] * 2,
        )
        self.assertEqual(self.job_manager.jobs()[-1].status, "completed")

    def test_paid_whole_video_analysis_displays_and_confirms_total_cost(
        self,
    ) -> None:
        video_path = Path(self.temporary_directory.name) / "paid-height.avi"
        _write_test_video(video_path, frame_count=121)
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))
        self.workspace.sync_shared_controls()
        self.workspace.infer_ceiling_height_button.click()
        dialog = self.workspace._ceiling_height_dialog
        self.assertIsNotNone(dialog)
        assert dialog is not None
        dialog.frame_interval_spinbox.setValue(1)

        button_text = dialog.analysis_button.text()
        self.assertRegex(
            button_text,
            r"Analyse video's ceiling height \(estimated .+\)",
        )
        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            dialog.analysis_button.click()

        question.assert_called_once()
        confirmation_text = str(question.call_args.args[2])
        self.assertIn("121 paid vision requests", confirmation_text)
        self.assertRegex(confirmation_text, r"\(estimated .+\)")
        self.assertIsNone(self.workspace._ceiling_height_runtime)
        self.assertEqual(self.job_manager.jobs(), ())

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

    def test_controls_below_seekbar_have_ten_pixel_padding(self) -> None:
        controls_layout = self.workspace.controls_row.layout()

        self.assertIsNotNone(controls_layout)
        assert controls_layout is not None
        self.assertEqual(
            controls_layout.getContentsMargins(),
            (GENERATION_CONTROLS_PADDING,) * 4,
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
            controls_layout.stretch(1),
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
        division_options = self.workspace.findChild(
            QWidget,
            "merged_generation_division_options_section",
        )
        profile_options = self.workspace.findChild(
            QWidget,
            "merged_generation_opening_profile_options_section",
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
        self.assertIsNotNone(division_options)
        self.assertIsNotNone(profile_options)
        self.assertIsNotNone(generation_actions)
        self.assertIsNotNone(face_actions)
        self.assertIsNotNone(editing_section)
        self.assertIsNotNone(creation_section)
        assert primary_actions is not None
        assert primary_column is not None
        assert generation_settings is not None
        assert division_options is not None
        assert profile_options is not None
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
            self.workspace.generate_texture_stack,
        )
        self.assertEqual(self.workspace.generate_texture_stack.count(), 2)
        self.assertIs(
            self.workspace.generate_texture_stack.widget(0),
            self.objects.generate_texture_button,
        )
        self.assertIs(
            self.workspace.generate_texture_stack.widget(1),
            self.surfaces.generate_button,
        )
        settings_layout = generation_settings.layout()
        self.assertIs(settings_layout.itemAt(0).widget(), division_options)
        self.assertIs(settings_layout.itemAt(1).widget(), profile_options)
        self.assertIs(
            settings_layout.itemAt(3).widget(),
            self.objects.meshy_target_polycount_control,
        )
        self.assertTrue(
            division_options.isAncestorOf(
                self.objects.symmetric_division_checkbox
            )
        )
        self.assertTrue(
            division_options.isAncestorOf(
                self.objects.side_door_duplication_checkbox
            )
        )
        self.assertTrue(
            profile_options.isAncestorOf(
                self.objects.generate_displacement_checkbox
            )
        )
        self.assertTrue(
            profile_options.isAncestorOf(
                self.objects.preserve_door_profile_checkbox
            )
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
            self.objects.side_door_duplication_checkbox,
            self.objects.generate_displacement_checkbox,
            self.objects.preserve_door_profile_checkbox,
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
        generation_header = creation_layout.itemAt(0).widget()
        self.assertIsNotNone(generation_header)
        assert generation_header is not None
        self.assertTrue(
            generation_header.isAncestorOf(
                self.workspace.object_generation_heading_label
            )
        )
        self.assertTrue(
            generation_header.isAncestorOf(
                self.workspace.door_slot_generation_indicator
            )
        )
        self.assertIs(creation_layout.itemAt(1).widget(), generation_settings)
        self.assertIs(
            creation_layout.itemAt(2).widget(),
            self.workspace.texture_provider_control,
        )
        self.assertIs(creation_layout.itemAt(3).widget(), primary_actions)
        self.assertIs(creation_layout.itemAt(4).widget(), generation_actions)

    def test_surface_controls_are_folded_into_contextual_object_actions(
        self,
    ) -> None:
        self.assertIsNone(
            self.workspace.findChild(
                QWidget,
                "merged_generation_surface_controls",
            )
        )
        self.assertTrue(
            self.workspace.object_creation_section.isAncestorOf(
                self.surfaces.surface_texture_provider_combo
            )
        )
        provider_label = self.workspace.texture_provider_control.findChild(QLabel)
        self.assertIsNotNone(provider_label)
        assert provider_label is not None
        self.assertEqual(provider_label.text(), "Texture provider")

        reference = np.full((8, 8, 4), 255, dtype=np.uint8)
        settings = replace(
            self.objects.get_runtime_settings(),
            meshy_api_key="meshy-test-key",
        )
        self.objects.set_runtime_settings(settings)
        self.surfaces.set_runtime_settings(settings)
        self.objects.set_temporary_object_reference(reference)
        self.surfaces.set_temporary_reference(reference)
        self.workspace.sync_shared_controls()
        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.objects.generate_texture_button,
        )
        self.assertTrue(self.objects.generate_button.isEnabled())
        self.assertTrue(self.objects.generate_geometry_button.isEnabled())

        mesh = trimesh.creation.box(extents=(1.0, 0.1, 1.0))
        surface = FixedSurface(
            surface_id="level:0/room:0/wall:1:2",
            surface_type="wall",
            level_index=0,
            room_index=0,
            mesh=mesh,
            area_square_meters=float(mesh.area),
        )
        self.surfaces.set_scene_surface_selection((surface,))

        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.surfaces.generate_button,
        )
        self.assertTrue(self.surfaces.generate_button.isEnabled())
        self.assertFalse(self.objects.generate_button.isEnabled())
        self.assertFalse(self.objects.generate_geometry_button.isEnabled())

        self.surfaces.set_scene_surface_selection(())

        self.assertIs(
            self.workspace.generate_texture_stack.currentWidget(),
            self.objects.generate_texture_button,
        )
        self.assertTrue(self.objects.generate_button.isEnabled())
        self.assertTrue(self.objects.generate_geometry_button.isEnabled())

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
        self.assertEqual(material_layout.count(), 2)
        self.assertEqual(
            material_layout.indexOf(self.objects.cancel_operation_button),
            -1,
        )

    def test_opening_generation_sections_follow_the_active_slot(self) -> None:
        division = self.workspace.division_options_section
        profile = self.workspace.opening_profile_options_section

        self.assertFalse(division.isHidden())
        self.assertTrue(profile.isHidden())

        self.assertTrue(
            self.objects.set_door_slot_editing_target(
                door_id="door-1",
                slot_id=DOOR_SLOT_BODY,
                object_id="door-body",
                object_name="Body",
                slot_display_name="Body",
            )
        )
        self.assertFalse(division.isHidden())
        self.assertFalse(profile.isHidden())

        self.assertTrue(
            self.objects.set_door_slot_editing_target(
                door_id="door-1",
                slot_id=DOOR_SLOT_KNOB,
                object_id="door-handle",
                object_name="Handle",
                slot_display_name="Handle",
            )
        )
        self.assertTrue(division.isHidden())
        self.assertTrue(profile.isHidden())

        self.objects.clear_door_slot_editing_target()
        self.assertFalse(division.isHidden())
        self.assertTrue(profile.isHidden())

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

    def test_seek_exits_opening_slot_only_when_the_frame_changes(self) -> None:
        video_path = Path(self.temporary_directory.name) / "source.avi"
        _write_test_video(video_path)
        self.objects.load_video(str(video_path))
        self.surfaces.load_video(str(video_path))
        self.assertTrue(
            self.objects.set_door_slot_editing_target(
                door_id="door-1",
                slot_id=DOOR_SLOT_BODY,
                object_id="door-body",
                object_name="Body",
                slot_display_name="Body",
            )
        )

        self.workspace._handle_seekbar_changed(0)
        self.assertIsNotNone(self.objects._door_slot_editing_target)

        self.workspace._handle_seekbar_changed(1)
        self.assertIsNone(self.objects._door_slot_editing_target)
        self.assertTrue(
            self.workspace.opening_profile_options_section.isHidden()
        )

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


# ### Generation action cancellation tests ###
class GenerationActionCancellationUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.job_manager = GenerationJobManager()
        self.objects = _ActionJobGenerationWorkspace(
            asset_directory=root / "objects",
            job_manager=self.job_manager,
        )
        self.surfaces = _ActionJobSurfaceWorkspace(
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

    def _enable_object_actions(self) -> None:
        self.objects.generate_geometry_button.setEnabled(True)
        self.objects.generate_texture_button.setEnabled(True)
        self.objects.generate_button.setEnabled(True)

    def test_object_action_buttons_cancel_only_their_own_launch(self) -> None:
        self._enable_object_actions()

        self.objects.generate_geometry_button.click()
        geometry_job_id = self.objects.action_job_ids["geometry"][0]
        self.assertEqual(self.objects.generate_geometry_button.text(), "Cancel")
        self.assertEqual(self.objects.generate_button.text(), "Generate")
        self.objects._sync_controls()
        self.assertTrue(self.objects.generate_geometry_button.isEnabled())
        self.objects.generate_button.setEnabled(True)

        self.objects.generate_button.click()
        generate_job_ids = self.objects.action_job_ids["generate"]
        self.assertEqual(len(generate_job_ids), 2)
        self.assertEqual(self.objects.generate_button.text(), "Cancel")
        self.assertEqual(self.objects.generate_geometry_button.text(), "Cancel")

        job_count = len(self.job_manager.jobs())
        self.objects.generate_geometry_button.click()

        self.assertEqual(len(self.job_manager.jobs()), job_count)
        self.assertEqual(self.objects.cancelled_actions, ["geometry:1"])
        self.assertEqual(
            self.job_manager.get_job(geometry_job_id).status,
            JOB_STATUS_CANCELLING,
        )
        self.assertTrue(
            all(
                self.job_manager.get_job(job_id).status == "running"
                for job_id in generate_job_ids
            )
        )
        self.assertEqual(self.objects.generate_geometry_button.text(), "Cancel")
        self.assertEqual(self.objects.generate_button.text(), "Cancel")

        self.job_manager.mark_cancelled(geometry_job_id)
        self.assertEqual(
            self.objects.generate_geometry_button.text(),
            "Generate geometry",
        )
        self.assertEqual(self.objects.generate_button.text(), "Cancel")

        self.objects.generate_button.click()
        self.assertEqual(
            self.objects.cancelled_actions,
            ["geometry:1", "generate:1", "generate:2"],
        )
        for job_id in generate_job_ids:
            self.assertEqual(
                self.job_manager.get_job(job_id).status,
                JOB_STATUS_CANCELLING,
            )
            self.job_manager.mark_cancelled(job_id)
        self.assertEqual(self.objects.generate_button.text(), "Generate")

    def test_object_texture_button_restores_after_completion(self) -> None:
        self._enable_object_actions()

        self.objects.generate_texture_button.click()
        texture_job_id = self.objects.action_job_ids["object_texture"][0]

        self.assertEqual(self.objects.generate_texture_button.text(), "Cancel")
        self.job_manager.complete_job(texture_job_id)
        self.assertEqual(
            self.objects.generate_texture_button.text(),
            "Generate texture",
        )
        self.assertEqual(self.objects.cancelled_actions, [])

    def test_surface_texture_cancel_is_isolated_from_object_texture(self) -> None:
        self._enable_object_actions()
        self.surfaces.generate_button.setEnabled(True)
        self.workspace.generate_texture_stack.setCurrentWidget(
            self.surfaces.generate_button
        )

        self.surfaces.generate_button.click()
        surface_job_id = self.surfaces.action_job_ids[0]
        self.workspace.generate_texture_stack.setCurrentWidget(
            self.objects.generate_texture_button
        )
        self.objects.generate_texture_button.click()
        object_job_id = self.objects.action_job_ids["object_texture"][0]

        self.assertEqual(self.surfaces.generate_button.text(), "Cancel")
        self.assertEqual(self.objects.generate_texture_button.text(), "Cancel")
        self.surfaces._sync_controls()
        self.assertTrue(self.surfaces.generate_button.isEnabled())

        self.surfaces.generate_button.click()

        self.assertEqual(
            self.surfaces.cancelled_actions,
            ["surface_texture"],
        )
        self.assertEqual(self.objects.cancelled_actions, [])
        self.assertEqual(
            self.job_manager.get_job(surface_job_id).status,
            JOB_STATUS_CANCELLING,
        )
        self.assertEqual(
            self.job_manager.get_job(object_job_id).status,
            "running",
        )
        self.assertEqual(self.objects.generate_texture_button.text(), "Cancel")

        self.job_manager.mark_cancelled(surface_job_id)
        self.job_manager.complete_job(object_job_id)
        self.assertEqual(
            self.surfaces.generate_button.text(),
            "Generate texture",
        )
        self.assertEqual(
            self.objects.generate_texture_button.text(),
            "Generate texture",
        )

    def test_failed_action_validation_does_not_turn_button_into_cancel(self) -> None:
        self._enable_object_actions()
        self.objects.skip_next_action = "geometry"

        self.objects.generate_geometry_button.click()

        self.assertEqual(self.objects.action_job_ids, {})
        self.assertEqual(self.objects.generate_geometry_button.text(), "Generate geometry")


if __name__ == "__main__":
    unittest.main()
