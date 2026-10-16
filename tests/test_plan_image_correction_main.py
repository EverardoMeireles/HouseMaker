# ### Environment setup ###
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ### Imports ###
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from PySide6.QtWidgets import QApplication

from housemaker.app_settings import ApplicationSettingsStore
from housemaker.canvas_plan_alignment import (
    CanvasPlanAlignmentResult,
    CanvasPlanAlignmentTransform,
)
from housemaker.generation_jobs import (
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
)
from housemaker.main import (
    BlueprintWorkspace,
    _build_local_file_revision,
    _PlanImageCorrectionRuntime,
    _PlanImageCorrectionThread,
)
from housemaker.models import (
    EditableSurfaceMeshData,
    EditableSurfaceVertexData,
    OpenSpaceData,
    StairData,
)
from housemaker.plan_correction_models import (
    PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
    PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
    PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
    PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
    plan_correction_model_label,
)
from housemaker.plan_image_correction import (
    PlanCorrectionProgress,
    PlanImageCorrectionResult,
)
from housemaker.plan_metric_state import (
    PlanMeasurementEvidence,
    PlanMetricCalibration,
)

# ### Module state ###
_qt_application = QApplication.instance() or QApplication([])
_qt_application.setQuitOnLastWindowClosed(False)


# ### Test helpers ###
def _plan_metric_calibration() -> PlanMetricCalibration:
    measurements = tuple(
        PlanMeasurementEvidence(
            label=label,
            value_meters=value_meters,
            start_normalized=start,
            end_normalized=end,
            corrected_start_normalized=start,
            corrected_end_normalized=end,
            confidence="high",
        )
        for label, value_meters, start, end in (
            ("610", 6.10, (0.10, 0.20), (0.90, 0.20)),
            ("390", 3.90, (0.20, 0.10), (0.20, 0.80)),
        )
    )
    return PlanMetricCalibration(
        estimated_meters_per_pixel=0.01,
        target_meters_per_pixel=0.02,
        normalization_scale=0.5,
        relative_residual=0.015,
        confidence="high",
        source_image_size=(2000, 1000),
        pre_normalization_size=(2000, 1000),
        normalized_image_size=(1000, 500),
        measurements=measurements,
    )


# ### Test doubles ###
class _FakeSignal:
    def __init__(self) -> None:
        self._slots: list[object] = []

    def connect(self, slot: object) -> None:
        self._slots.append(slot)

    def emit(self, value: object | None = None) -> None:
        for slot in tuple(self._slots):
            if value is None:
                slot()  # type: ignore[operator]
            else:
                slot(value)  # type: ignore[operator]


class _FakeCorrectionThread:
    """Controllable worker replacement that never invokes the correction model."""

    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        api_key: str,
        model: str = PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
        parent: object | None = None,
        *,
        make_walls_continuous: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.api_key = api_key
        self.model = model
        self.parent = parent
        del make_walls_continuous
        self.progress = _FakeSignal()
        self.finished = _FakeSignal()
        self.result: PlanImageCorrectionResult | None = None
        self.error_message: str | None = None
        self.was_cancelled = False
        self.running = False
        self.interruption_requested = False
        self.deleted_later = False
        self.metric_normalizer: object | None = None
        self.alignment_coordinator: object | None = None
        self.alignment_level_index: int | None = None
        self.alignment_outcome: object | None = None

    def set_metric_normalizer(self, metric_normalizer: object) -> None:
        self.metric_normalizer = metric_normalizer

    def set_alignment_coordinator(
        self,
        alignment_coordinator: object,
        level_index: int,
    ) -> None:
        self.alignment_coordinator = alignment_coordinator
        self.alignment_level_index = int(level_index)

    def start(self) -> None:
        self.running = True

    def finish(self) -> None:
        self.running = False
        if (
            self.result is not None
            and self.error_message is None
            and not self.was_cancelled
            and self.alignment_coordinator is not None
        ):
            assert self.alignment_level_index is not None
            self.alignment_outcome = self.alignment_coordinator.submit(  # type: ignore[attr-defined]
                self.alignment_level_index,
                self.result.output_path,
            )
        self.finished.emit()

    def isRunning(self) -> bool:
        return self.running

    def requestInterruption(self) -> None:
        self.interruption_requested = True

    def wait(self, _milliseconds: int) -> bool:
        self.running = False
        return True

    def deleteLater(self) -> None:
        self.deleted_later = True


# ### Main-workspace integration tests ###
class PlanImageCorrectionMainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._openai_environment_patch = patch.dict(
            os.environ,
            {"OPENAI_API_KEY": ""},
        )
        self._openai_environment_patch.start()
        self.addCleanup(self._openai_environment_patch.stop)
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_path = Path(self._temporary_directory.name)
        self.workspace = BlueprintWorkspace(
            application_settings=ApplicationSettingsStore(
                self.temporary_path / "settings.json"
            )
        )
        self.workspace.resize(1200, 800)
        self.workspace.show()
        self.workspace.settings_widget.openai_api_key_edit.setText(
            "openai-plan-test-key"
        )
        _qt_application.processEvents()

    def tearDown(self) -> None:
        self.workspace.shutdown()
        self.workspace.close()
        self.workspace.deleteLater()
        _qt_application.processEvents()
        self._temporary_directory.cleanup()

    def _load_source_image(
        self,
        name: str = "folded-plan.jpg",
        *,
        size: tuple[int, int] = (96, 64),
    ) -> Path:
        source_path = self.temporary_path / name
        Image.new("RGB", size, "white").save(source_path)
        self.workspace._set_current_level_image(str(source_path))
        return source_path.resolve()

    def _load_level_source_image(
        self,
        level_index: int,
        *,
        name: str | None = None,
        size: tuple[int, int] = (96, 64),
    ) -> Path:
        """Attach a valid original plan photo to one arbitrary project level."""

        level = self.workspace.levels[level_index]
        source_path = self.temporary_path / (
            name or f"folded-plan-L{level.index}.jpg"
        )
        Image.new("RGB", size, "white").save(source_path)
        resolved_path = source_path.resolve()
        level.image_path = str(resolved_path)
        level.original_image_path = str(resolved_path)
        level.image_size_pixels = tuple(float(value) for value in size)
        return resolved_path

    def _start_correction(self) -> tuple[_FakeCorrectionThread, str]:
        with patch(
            "housemaker.main._PlanImageCorrectionThread",
            _FakeCorrectionThread,
        ):
            self.workspace._handle_api_image_correction_clicked()

        runtime = self.workspace._plan_image_correction_runtimes[
            self.workspace.current_level.index
        ]
        thread = runtime.thread
        self.assertIsInstance(thread, _FakeCorrectionThread)
        return thread, runtime.job_id  # type: ignore[return-value]

    @staticmethod
    def _successful_result(
        thread: _FakeCorrectionThread,
        *,
        size: tuple[int, int] = (80, 50),
        metric_calibration: PlanMetricCalibration | None = None,
        metric_warning: str = "",
    ) -> PlanImageCorrectionResult:
        thread.output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("1", size, 1).save(thread.output_path)
        return PlanImageCorrectionResult(
            output_path=thread.output_path,
            method="openai",
            input_size=(96, 64),
            output_size=size,
            stages=("loading", "ai_correction", "complete"),
            metric_calibration=metric_calibration,
            metric_warning=metric_warning,
        )

    def test_plan_image_buttons_and_source_state(self) -> None:
        self.assertEqual(self.workspace.load_image_button.text(), "Load plan image")
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Image correction",
        )
        # The chooser remains available to import JSON before loading images.
        self.assertTrue(self.workspace.image_correction_button.isEnabled())
        self.assertIn(
            "OpenAI models",
            self.workspace.image_correction_button.toolTip(),
        )
        self.assertIn(
            "Qwen Research License",
            self.workspace.image_correction_button.toolTip(),
        )

        source_path = self._load_source_image()

        self.assertTrue(self.workspace.image_correction_button.isEnabled())
        self.assertEqual(self.workspace.current_level.image_path, str(source_path))
        self.assertEqual(
            self.workspace.current_level.original_image_path,
            str(source_path),
        )

    def test_correction_is_available_when_only_another_level_has_a_plan(
        self,
    ) -> None:
        other_level = next(
            level
            for level in self.workspace.levels
            if level.index != self.workspace.current_level.index
        )
        self._load_level_source_image(other_level.index)

        self.workspace._update_image_correction_button_state()

        self.assertTrue(self.workspace.image_correction_button.isEnabled())
        with patch(
            "housemaker.main._PlanImageCorrectionThread",
            _FakeCorrectionThread,
        ):
            self.workspace._handle_api_image_correction_clicked()
        self.assertEqual(
            set(self.workspace._plan_image_correction_runtimes),
            {other_level.index},
        )

    def test_worker_forwards_the_selected_openai_model_and_key(self) -> None:
        source_path = self._load_source_image()
        output_path = self.temporary_path / "corrected.png"
        expected_result = PlanImageCorrectionResult(
            output_path=output_path,
            method="openai",
            input_size=(96, 64),
            output_size=(96, 64),
            stages=("loading", "openai_correction", "complete"),
        )
        for model in (
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        ):
            with self.subTest(model=model):
                worker = _PlanImageCorrectionThread(
                    source_path,
                    output_path,
                    "configured-openai-key",
                    model,
                    make_walls_continuous=True,
                )

                analyzer = object()
                with (
                    patch(
                        "housemaker.main.create_plan_measurement_analyzer",
                        return_value=analyzer,
                    ) as create_analyzer,
                    patch(
                        "housemaker.main.correct_plan_image",
                        return_value=expected_result,
                    ) as correct,
                ):
                    worker.run()
                
                self.assertIs(worker.result, expected_result)
                self.assertEqual(worker.model, model)
                correct.assert_called_once()
                self.assertEqual(
                    correct.call_args.args,
                    (source_path, output_path),
                )
                self.assertEqual(
                    correct.call_args.kwargs["api_key"],
                    "configured-openai-key",
                )
                self.assertEqual(correct.call_args.kwargs["model"], model)
                self.assertTrue(
                    correct.call_args.kwargs["make_walls_continuous"]
                )
                self.assertIs(
                    correct.call_args.kwargs["measurement_analyzer"],
                    analyzer,
                )
                create_analyzer.assert_called_once_with(
                    model,
                    api_key="configured-openai-key",
                )
                worker.deleteLater()

    def test_worker_uses_batch_normalizer_without_single_plan_analysis(
        self,
    ) -> None:
        source_path = self._load_source_image()
        output_path = self.temporary_path / "batch-corrected.png"
        expected_result = PlanImageCorrectionResult(
            output_path=output_path,
            method="openai",
            input_size=(96, 64),
            output_size=(96, 64),
            stages=("loading", "ai_correction", "complete"),
        )
        normalizer = Mock()
        worker = _PlanImageCorrectionThread(
            source_path,
            output_path,
            "configured-openai-key",
        )
        worker.set_metric_normalizer(normalizer)

        with (
            patch(
                "housemaker.main.create_plan_measurement_analyzer"
            ) as create_analyzer,
            patch(
                "housemaker.main.correct_plan_image",
                return_value=expected_result,
            ) as correct,
        ):
            worker.run()

        create_analyzer.assert_not_called()
        self.assertIs(
            correct.call_args.kwargs["metric_normalizer"],
            normalizer,
        )
        self.assertIsNone(correct.call_args.kwargs["measurement_analyzer"])
        worker.deleteLater()

    def test_successful_async_correction_applies_result_and_preserves_source(
        self,
    ) -> None:
        source_path = self._load_source_image()
        thread, job_id = self._start_correction()

        self.assertEqual(thread.input_path, source_path)
        self.assertEqual(thread.api_key, "openai-plan-test-key")
        self.assertTrue(thread.running)
        self.assertFalse(self.workspace.image_correction_button.isEnabled())
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Correcting...",
        )

        thread.progress.emit(
            PlanCorrectionProgress(
                stage="binarization",
                percent=88,
                message="Creating black-and-white plan",
            )
        )
        active_job = self.workspace.job_manager.get_job(job_id)
        self.assertIsNotNone(active_job)
        self.assertEqual(active_job.progress, 88)  # type: ignore[union-attr]
        self.assertEqual(
            active_job.stage,  # type: ignore[union-attr]
            "Creating black-and-white plan (88%)",
        )

        thread.result = self._successful_result(thread)
        thread.finish()

        level = self.workspace.current_level
        self.assertEqual(level.image_path, str(thread.output_path.resolve()))
        self.assertEqual(level.original_image_path, str(source_path))
        self.assertEqual(level.image_size_pixels, (80.0, 50.0))
        self.assertIn(
            "corrected from folded-plan.jpg", self.workspace.blueprint_name_label.text()
        )
        self.assertIn(
            "metric scale uncalibrated",
            self.workspace.blueprint_name_label.text(),
        )
        self.assertIsNone(level.plan_metric_calibration)
        self.assertNotIn(level.index, self.workspace._plan_image_correction_runtimes)
        self.assertTrue(self.workspace.image_correction_button.isEnabled())
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Image correction",
        )
        completed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(
            completed_job.status,  # type: ignore[union-attr]
            JOB_STATUS_COMPLETED,
        )
        self.assertEqual(
            completed_job.stage,  # type: ignore[union-attr]
            f"{plan_correction_model_label(PLAN_CORRECTION_MODEL_GPT_IMAGE_2)} "
            "plan correction completed; metric scale uncalibrated",
        )

        second_thread, _second_job_id = self._start_correction()
        self.assertEqual(second_thread.input_path, source_path)
        self.assertNotEqual(second_thread.input_path, thread.output_path)

    def test_one_click_starts_every_loaded_plan_correction(self) -> None:
        loaded_sources = {
            level_index: self._load_level_source_image(level_index)
            for level_index in (1, 2, 3)
        }

        with patch(
            "housemaker.main._PlanImageCorrectionThread",
            _FakeCorrectionThread,
        ):
            self.workspace._handle_api_image_correction_clicked()

        self.assertEqual(
            set(self.workspace._plan_image_correction_runtimes),
            set(loaded_sources),
        )
        self.assertEqual(len(self.workspace.job_manager.jobs()), 3)
        for level_index, source_path in loaded_sources.items():
            with self.subTest(level_index=level_index):
                runtime = self.workspace._plan_image_correction_runtimes[
                    level_index
                ]
                thread = runtime.thread
                self.assertIsInstance(thread, _FakeCorrectionThread)
                self.assertEqual(thread.input_path, source_path)  # type: ignore[attr-defined]
                self.assertTrue(thread.running)  # type: ignore[attr-defined]
                self.assertTrue(callable(thread.metric_normalizer))
                self.assertIn(
                    f"corrected-plan-L{level_index}-",
                    runtime.output_path.name,
                )

        unloaded_level_indices = {
            level.index
            for level in self.workspace.levels
            if level.index not in loaded_sources
        }
        self.assertTrue(unloaded_level_indices)
        self.assertTrue(
            unloaded_level_indices.isdisjoint(
                self.workspace._plan_image_correction_runtimes
            )
        )
        self.assertFalse(self.workspace.image_correction_button.isEnabled())
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Correcting...",
        )

    def test_second_click_does_not_duplicate_an_active_batch(self) -> None:
        self._load_level_source_image(1)
        self._load_level_source_image(2)

        with patch(
            "housemaker.main._PlanImageCorrectionThread",
            _FakeCorrectionThread,
        ):
            self.workspace._handle_api_image_correction_clicked()
            runtimes_before_second_click = dict(
                self.workspace._plan_image_correction_runtimes
            )
            job_ids_before_second_click = tuple(
                job.job_id for job in self.workspace.job_manager.jobs()
            )

            self.workspace._handle_api_image_correction_clicked()

        self.assertEqual(
            self.workspace._plan_image_correction_runtimes,
            runtimes_before_second_click,
        )
        self.assertEqual(
            tuple(job.job_id for job in self.workspace.job_manager.jobs()),
            job_ids_before_second_click,
        )

    def test_no_loaded_plan_is_a_warning_only_noop(self) -> None:
        with (
            patch("housemaker.main._PlanImageCorrectionThread") as thread_type,
            patch("housemaker.main.QMessageBox.warning") as warning,
        ):
            self.workspace._handle_api_image_correction_clicked()

        thread_type.assert_not_called()
        warning.assert_called_once_with(
            self.workspace,
            "Image correction unavailable",
            "Load a valid plan image before correcting it.",
        )
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})
        self.assertEqual(self.workspace.job_manager.jobs(), ())

    def test_successful_batch_is_applied_together_after_every_result(self) -> None:
        source_paths = {
            level_index: self._load_level_source_image(level_index)
            for level_index in (2, 3)
        }
        with patch(
            "housemaker.main._PlanImageCorrectionThread",
            _FakeCorrectionThread,
        ):
            self.workspace._handle_api_image_correction_clicked()

        runtimes = dict(self.workspace._plan_image_correction_runtimes)
        first_runtime = runtimes[2]
        first_thread = first_runtime.thread
        self.assertIsInstance(first_thread, _FakeCorrectionThread)
        first_thread.result = self._successful_result(  # type: ignore[attr-defined]
            first_thread,  # type: ignore[arg-type]
        )
        first_thread.finish()  # type: ignore[attr-defined]

        for level_index, source_path in source_paths.items():
            self.assertEqual(
                self.workspace.levels[level_index].image_path,
                str(source_path),
            )
        self.assertTrue(first_runtime.output_path.is_file())
        self.assertFalse(self.workspace.image_correction_button.isEnabled())
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Correcting...",
        )

        second_runtime = runtimes[3]
        second_thread = second_runtime.thread
        self.assertIsInstance(second_thread, _FakeCorrectionThread)
        second_thread.result = self._successful_result(  # type: ignore[attr-defined]
            second_thread,  # type: ignore[arg-type]
        )
        second_thread.finish()  # type: ignore[attr-defined]

        for level_index, runtime in runtimes.items():
            with self.subTest(level_index=level_index):
                self.assertEqual(
                    self.workspace.levels[level_index].image_path,
                    str(runtime.output_path.resolve()),
                )
                self.assertEqual(
                    self.workspace.levels[level_index].original_image_path,
                    str(source_paths[level_index]),
                )
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})
        self.assertTrue(self.workspace.image_correction_button.isEnabled())
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Image correction",
        )

    def test_successful_batch_alignment_changes_only_canvas_offsets(
        self,
    ) -> None:
        for level_index in (2, 3):
            self._load_level_source_image(level_index)
        ground = self.workspace.levels[2]
        story = self.workspace.levels[3]
        ground.canvas_level_scale = 1.4
        ground.canvas_offset_x_pixels = 90.0
        story.canvas_level_scale = 0.7
        story.canvas_offset_y_pixels = -55.0
        with patch(
            "housemaker.main._PlanImageCorrectionThread",
            _FakeCorrectionThread,
        ):
            self.workspace._handle_api_image_correction_clicked()

        runtimes = dict(self.workspace._plan_image_correction_runtimes)
        calibration = _plan_metric_calibration()
        alignment = CanvasPlanAlignmentResult(
            anchor_level_index=2,
            transforms=(
                CanvasPlanAlignmentTransform(
                    level_index=2,
                    canvas_offset_x_pixels=90.0,
                    canvas_offset_y_pixels=0.0,
                    reference_level_index=2,
                    score=1.0,
                    wall_score=1.0,
                    stair_score=0.0,
                ),
                CanvasPlanAlignmentTransform(
                    level_index=3,
                    canvas_offset_x_pixels=14.0,
                    canvas_offset_y_pixels=-22.0,
                    reference_level_index=2,
                    score=0.88,
                    wall_score=0.86,
                    stair_score=0.2,
                ),
            ),
        )
        with patch(
            "housemaker.plan_correction_alignment.align_canvas_plan_levels",
            return_value=alignment,
        ) as align:
            for level_index in (2, 3):
                thread = runtimes[level_index].thread
                thread.result = self._successful_result(  # type: ignore[attr-defined]
                    thread,  # type: ignore[arg-type]
                    size=calibration.normalized_image_size,
                    metric_calibration=calibration,
                )
                thread.finish()  # type: ignore[attr-defined]

        align.assert_called_once()
        self.assertNotIn("lock_native_scale", align.call_args.kwargs)
        self.assertEqual(ground.canvas_level_scale, 1.4)
        self.assertEqual(ground.canvas_offset_x_pixels, 90.0)
        self.assertEqual(ground.canvas_offset_y_pixels, 0.0)
        self.assertEqual(story.canvas_level_scale, 0.7)
        self.assertEqual(story.canvas_offset_x_pixels, 14.0)
        self.assertEqual(story.canvas_offset_y_pixels, -22.0)
        self.assertEqual(self.workspace.canvas.canvas_level_scale, 1.4)
        self.assertEqual(self.workspace.canvas.canvas_offset_x_pixels, 90.0)
        self.assertEqual(self.workspace.canvas.canvas_offset_y_pixels, 0.0)

    def test_failed_batch_discards_staged_outputs_and_preserves_every_source(
        self,
    ) -> None:
        source_paths = {
            level_index: self._load_level_source_image(level_index)
            for level_index in (2, 3)
        }
        with patch(
            "housemaker.main._PlanImageCorrectionThread",
            _FakeCorrectionThread,
        ):
            self.workspace._handle_api_image_correction_clicked()

        runtimes = dict(self.workspace._plan_image_correction_runtimes)
        staged_runtime = runtimes[2]
        staged_thread = staged_runtime.thread
        self.assertIsInstance(staged_thread, _FakeCorrectionThread)
        staged_thread.result = self._successful_result(  # type: ignore[attr-defined]
            staged_thread,  # type: ignore[arg-type]
        )
        staged_thread.finish()  # type: ignore[attr-defined]
        self.assertTrue(staged_runtime.output_path.is_file())

        failed_runtime = runtimes[3]
        failed_thread = failed_runtime.thread
        self.assertIsInstance(failed_thread, _FakeCorrectionThread)
        failure_reason = "The plan correction provider rejected the image."
        failed_thread.error_message = failure_reason  # type: ignore[attr-defined]
        with patch("housemaker.main.QMessageBox.critical") as critical:
            failed_thread.finish()  # type: ignore[attr-defined]

        for level_index, runtime in runtimes.items():
            with self.subTest(level_index=level_index):
                self.assertEqual(
                    self.workspace.levels[level_index].image_path,
                    str(source_paths[level_index]),
                )
                self.assertEqual(
                    self.workspace.levels[level_index].original_image_path,
                    str(source_paths[level_index]),
                )
                self.assertFalse(runtime.output_path.exists())
                job = self.workspace.job_manager.get_job(runtime.job_id)
                self.assertEqual(
                    job.status,  # type: ignore[union-attr]
                    JOB_STATUS_FAILED,
                )
                if level_index == 3:
                    self.assertEqual(  # type: ignore[union-attr]
                        job.stage,
                        f"Failed: {failure_reason}",
                    )
                else:
                    self.assertEqual(  # type: ignore[union-attr]
                        job.stage,
                        "Failed: Image correction batch was not applied because "
                        "another plan failed.",
                    )
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})
        critical.assert_called_once_with(
            self.workspace,
            "Image correction failed",
            failure_reason,
        )

    def test_cancelling_batch_discards_every_output_and_preserves_sources(
        self,
    ) -> None:
        source_paths = {
            level_index: self._load_level_source_image(level_index)
            for level_index in (2, 3)
        }
        with patch(
            "housemaker.main._PlanImageCorrectionThread",
            _FakeCorrectionThread,
        ):
            self.workspace._handle_api_image_correction_clicked()

        runtimes = dict(self.workspace._plan_image_correction_runtimes)
        staged_runtime = runtimes[2]
        staged_thread = staged_runtime.thread
        self.assertIsInstance(staged_thread, _FakeCorrectionThread)
        staged_thread.result = self._successful_result(  # type: ignore[attr-defined]
            staged_thread,  # type: ignore[arg-type]
        )
        staged_thread.finish()  # type: ignore[attr-defined]
        self.assertTrue(staged_runtime.output_path.is_file())

        cancelled_runtime = runtimes[3]
        cancelled_thread = cancelled_runtime.thread
        self.assertIsInstance(cancelled_thread, _FakeCorrectionThread)
        self.assertTrue(
            self.workspace.job_manager.cancel_job(staged_runtime.job_id)
        )
        self.assertTrue(cancelled_thread.interruption_requested)  # type: ignore[attr-defined]
        cancelled_thread.result = self._successful_result(  # type: ignore[attr-defined]
            cancelled_thread,  # type: ignore[arg-type]
        )
        cancelled_thread.was_cancelled = True  # type: ignore[attr-defined]
        cancelled_thread.finish()  # type: ignore[attr-defined]

        for level_index, runtime in runtimes.items():
            with self.subTest(level_index=level_index):
                self.assertEqual(
                    self.workspace.levels[level_index].image_path,
                    str(source_paths[level_index]),
                )
                self.assertEqual(
                    self.workspace.levels[level_index].original_image_path,
                    str(source_paths[level_index]),
                )
                self.assertFalse(runtime.output_path.exists())
                self.assertEqual(
                    self.workspace.job_manager.get_job(
                        runtime.job_id
                    ).status,  # type: ignore[union-attr]
                    JOB_STATUS_CANCELLED,
                )
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})

    def test_luna_and_terra_selection_is_forwarded_and_labeled(self) -> None:
        self._load_source_image()
        combo = self.workspace.settings_widget.plan_correction_model_combo

        for model in (
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        ):
            with self.subTest(model=model):
                combo.setCurrentIndex(combo.findData(model))
                thread, job_id = self._start_correction()

                self.assertEqual(thread.model, model)
                self.assertEqual(thread.api_key, "openai-plan-test-key")

                thread.result = self._successful_result(thread)
                thread.finish()

                completed_job = self.workspace.job_manager.get_job(job_id)
                self.assertEqual(
                    completed_job.status,  # type: ignore[union-attr]
                    JOB_STATUS_COMPLETED,
                )
                self.assertEqual(
                    completed_job.stage,  # type: ignore[union-attr]
                    f"{plan_correction_model_label(model)} "
                    "plan correction completed; metric scale uncalibrated",
                )

    def test_successful_correction_applies_and_reports_metric_calibration(
        self,
    ) -> None:
        self._load_source_image()
        calibration = _plan_metric_calibration()
        self.workspace.current_level.scale = 2.25
        self.workspace.current_level.canvas_level_scale = 1.75
        self.workspace.current_level.canvas_offset_x_pixels = 125.0
        self.workspace.current_level.canvas_offset_y_pixels = -75.0
        thread, job_id = self._start_correction()
        thread.result = self._successful_result(
            thread,
            size=calibration.normalized_image_size,
            metric_calibration=calibration,
        )

        thread.finish()

        self.assertEqual(
            self.workspace.current_level.plan_metric_calibration,
            calibration,
        )
        self.assertEqual(self.workspace.current_level.scale, 1.0)
        self.assertEqual(self.workspace.current_level.canvas_level_scale, 1.75)
        self.assertEqual(
            self.workspace.current_level.canvas_offset_x_pixels,
            125.0,
        )
        self.assertEqual(
            self.workspace.current_level.canvas_offset_y_pixels,
            -75.0,
        )
        self.assertIn(
            "metric scale calibrated (high confidence), 50 px/m",
            self.workspace.blueprint_name_label.text(),
        )
        self.assertIn(
            "Accepted plan dimensions",
            self.workspace.blueprint_name_label.toolTip(),
        )
        self.assertIn("6.1 m", self.workspace.blueprint_name_label.toolTip())
        completed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(
            completed_job.stage,  # type: ignore[union-attr]
            f"{plan_correction_model_label(PLAN_CORRECTION_MODEL_GPT_IMAGE_2)} "
            "plan correction completed; metric scale calibrated "
            "(high confidence); 2 measurements accepted, 1.5% disagreement",
        )

    def test_loading_a_new_raw_plan_clears_metric_calibration(self) -> None:
        self._load_source_image()
        self.workspace.current_level.plan_metric_calibration = (
            _plan_metric_calibration()
        )
        replacement_path = self.temporary_path / "new-raw-plan.png"
        Image.new("RGB", (72, 48), "white").save(replacement_path)

        self.workspace._set_current_level_image(str(replacement_path))

        self.assertIsNone(
            self.workspace.current_level.plan_metric_calibration
        )
        self.assertIn(
            "metric scale uncalibrated",
            self.workspace.blueprint_name_label.text(),
        )

    def test_metric_failure_keeps_corrected_image_and_reports_warning(self) -> None:
        self._load_source_image()
        thread, job_id = self._start_correction()
        warning_text = "No two reliable dimensions agreed on scale."
        thread.result = self._successful_result(
            thread,
            metric_warning=warning_text,
        )

        with patch("housemaker.main.QMessageBox.warning") as warning:
            thread.finish()

        self.assertEqual(
            self.workspace.current_level.image_path,
            str(thread.output_path.resolve()),
        )
        self.assertTrue(thread.output_path.is_file())
        completed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(completed_job.status, JOB_STATUS_COMPLETED)
        self.assertIn(warning_text, completed_job.stage)
        warning.assert_called_once()
        self.assertIn(warning_text, warning.call_args.args[2])

    def test_geometry_guard_rejects_correction_before_worker_creation(self) -> None:
        self._load_source_image()
        self.workspace.current_level.vertex_data.add_vertex(10.0, 20.0)

        with (
            patch("housemaker.main._PlanImageCorrectionThread") as thread_type,
            patch("housemaker.main.QMessageBox.warning") as warning,
        ):
            self.workspace._handle_api_image_correction_clicked()

        thread_type.assert_not_called()
        warning.assert_called_once()
        self.assertEqual(
            warning.call_args.args[2],
            "Image correction would change the plan coordinate system. Remove "
            f"the following items from {self.workspace.current_level.display_name} "
            "before correcting it:\n\n"
            "- 1 wall vertex\n\n"
            "Reloading the plan image does not remove level geometry.",
        )
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})
        self.assertEqual(self.workspace.job_manager.jobs(), ())

    def test_geometry_guard_lists_retained_blocker_types_and_counts(self) -> None:
        self._load_source_image()
        level = self.workspace.current_level
        level.open_spaces.extend(
            (
                OpenSpaceData("opening-a", 10.0, 10.0, 30.0, 30.0),
                OpenSpaceData("opening-b", 40.0, 40.0, 70.0, 70.0),
            )
        )
        level.editable_surfaces.append(
            EditableSurfaceMeshData(
                source_surface_id=f"l{level.index}:floor",
                frame_kind="level_image",
                frame_normal_sign=1,
                vertices=(
                    EditableSurfaceVertexData(
                        vertex_id="1" * 32,
                        u=0.0,
                        v=0.0,
                        normal_offset_meters=0.0,
                    ),
                ),
                faces=(),
            )
        )
        self.workspace.stairs.extend(
            (
                StairData(
                    start_level_index=level.index,
                    start_a_x=0.0,
                    start_a_y=0.0,
                    start_b_x=10.0,
                    start_b_y=0.0,
                    end_level_index=level.index + 1,
                    end_a_x=0.0,
                    end_a_y=100.0,
                    end_b_x=10.0,
                    end_b_y=100.0,
                ),
                StairData(
                    start_level_index=level.index - 1,
                    start_a_x=20.0,
                    start_a_y=0.0,
                    start_b_x=30.0,
                    start_b_y=0.0,
                    end_level_index=level.index,
                    end_a_x=20.0,
                    end_a_y=100.0,
                    end_b_x=30.0,
                    end_b_y=100.0,
                ),
                StairData(
                    start_level_index=6,
                    start_a_x=40.0,
                    start_a_y=0.0,
                    start_b_x=50.0,
                    start_b_y=0.0,
                    end_level_index=7,
                    end_a_x=40.0,
                    end_a_y=100.0,
                    end_b_x=50.0,
                    end_b_y=100.0,
                ),
            )
        )

        with (
            patch("housemaker.main._PlanImageCorrectionThread") as thread_type,
            patch("housemaker.main.QMessageBox.warning") as warning,
        ):
            self.workspace._handle_api_image_correction_clicked()

        thread_type.assert_not_called()
        warning.assert_called_once_with(
            self.workspace,
            "Image correction unavailable",
            "Image correction would change the plan coordinate system. Remove "
            f"the following items from {level.display_name} before correcting it:\n\n"
            "- 2 open spaces\n"
            "- 1 editable surface\n"
            "- 2 connected stairs\n\n"
            "Reloading the plan image does not remove level geometry.",
        )
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})
        self.assertEqual(self.workspace.job_manager.jobs(), ())

    def test_batch_guard_groups_every_loaded_level_blocker_in_level_order(
        self,
    ) -> None:
        underground = self.workspace.levels[1]
        ground = self.workspace.levels[2]
        story = self.workspace.levels[3]
        self._load_level_source_image(underground.index)
        self._load_level_source_image(ground.index)
        self._load_level_source_image(story.index)
        underground.open_spaces.append(
            OpenSpaceData("underground-opening", 10.0, 10.0, 30.0, 30.0)
        )
        ground.vertex_data.add_vertex(10.0, 20.0)
        ground.vertex_data.add_vertex(40.0, 20.0)
        ground.vertex_data.add_edge(
            ground.vertex_data.vertices[0].id,
            ground.vertex_data.vertices[1].id,
        )
        story.open_spaces.extend(
            (
                OpenSpaceData("story-opening-a", 10.0, 10.0, 30.0, 30.0),
                OpenSpaceData("story-opening-b", 40.0, 40.0, 70.0, 70.0),
            )
        )

        with (
            patch("housemaker.main._PlanImageCorrectionThread") as thread_type,
            patch("housemaker.main.QMessageBox.warning") as warning,
        ):
            self.workspace._handle_api_image_correction_clicked()

        thread_type.assert_not_called()
        warning.assert_called_once_with(
            self.workspace,
            "Image correction unavailable",
            "Image correction would change the plan coordinate system. Remove "
            "the following items before correcting the loaded plans:\n\n"
            f"{underground.display_name}:\n"
            "- 1 open space\n\n"
            f"{ground.display_name}:\n"
            "- 2 wall vertices\n"
            "- 1 wall segment\n\n"
            f"{story.display_name}:\n"
            "- 2 open spaces\n\n"
            "Reloading a plan image does not remove level geometry.",
        )
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})
        self.assertEqual(self.workspace.job_manager.jobs(), ())

    def test_every_openai_model_requires_an_api_key(self) -> None:
        self._load_source_image()
        self.workspace.settings_widget.openai_api_key_edit.clear()
        output_directory = self.temporary_path / "corrected_plans"
        combo = self.workspace.settings_widget.plan_correction_model_combo

        for model in (
            PLAN_CORRECTION_MODEL_GPT_IMAGE_2,
            PLAN_CORRECTION_MODEL_GPT_5_6_LUNA,
            PLAN_CORRECTION_MODEL_GPT_5_6_TERRA,
        ):
            with (
                self.subTest(model=model),
                patch(
                    "housemaker.main._PlanImageCorrectionThread"
                ) as thread_type,
                patch("housemaker.main.QMessageBox.warning") as warning,
            ):
                combo.setCurrentIndex(combo.findData(model))
                self.workspace._handle_api_image_correction_clicked()

                thread_type.assert_not_called()
                warning.assert_called_once()
                self.assertEqual(
                    warning.call_args.args[1],
                    "OpenAI API key required",
                )
                self.assertIn("Settings", warning.call_args.args[2])
                self.assertEqual(
                    self.workspace._plan_image_correction_runtimes,
                    {},
                )
                self.assertEqual(self.workspace.job_manager.jobs(), ())
                self.assertFalse(output_directory.exists())

    def test_qwen_selection_does_not_require_an_openai_key(self) -> None:
        self._load_source_image()
        self.workspace.settings_widget.openai_api_key_edit.clear()
        combo = self.workspace.settings_widget.plan_correction_model_combo
        combo.setCurrentIndex(
            combo.findData(PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1)
        )

        with (
            patch(
                "housemaker.main.create_default_qwen_plan_image_editor"
            ) as create_qwen,
            patch(
                "housemaker.main._PlanImageCorrectionThread",
                _FakeCorrectionThread,
            ),
        ):
            self.workspace._handle_api_image_correction_clicked()

        create_qwen.assert_called_once_with()
        runtime = self.workspace._plan_image_correction_runtimes[
            self.workspace.current_level.index
        ]
        thread = runtime.thread
        self.assertEqual(thread.api_key, "")  # type: ignore[attr-defined]
        self.assertEqual(  # type: ignore[attr-defined]
            thread.model,
            PLAN_CORRECTION_MODEL_QWEN_IMAGE_2_1,
        )

    def test_failed_correction_reports_reason_and_preserves_source(self) -> None:
        source_path = self._load_source_image()
        thread, job_id = self._start_correction()
        thread.output_path.parent.mkdir(parents=True, exist_ok=True)
        thread.output_path.write_bytes(b"partial output")
        reason = "The local Qwen correction process failed."
        thread.error_message = reason

        with patch("housemaker.main.QMessageBox.critical") as critical:
            thread.finish()

        failed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(failed_job.status, JOB_STATUS_FAILED)  # type: ignore[union-attr]
        self.assertEqual(  # type: ignore[union-attr]
            failed_job.stage,
            f"Failed: {reason}",
        )
        self.assertEqual(self.workspace.current_level.image_path, str(source_path))
        self.assertEqual(
            self.workspace.current_level.original_image_path,
            str(source_path),
        )
        self.assertFalse(thread.output_path.exists())
        critical.assert_called_once()
        self.assertEqual(
            critical.call_args.args[1],
            "Image correction failed",
        )
        self.assertEqual(critical.call_args.args[2], reason)

    def test_changed_source_rejects_and_deletes_stale_result(self) -> None:
        source_path = self._load_source_image()
        thread, job_id = self._start_correction()
        thread.result = self._successful_result(thread)
        replacement_path = self.temporary_path / "replacement-plan.jpg"
        Image.new("RGB", (72, 48), "white").save(replacement_path)
        level = self.workspace.current_level
        level.image_path = str(replacement_path.resolve())
        level.original_image_path = str(replacement_path.resolve())

        with patch("housemaker.main.QMessageBox.warning") as warning:
            thread.finish()

        self.assertEqual(level.image_path, str(replacement_path.resolve()))
        self.assertEqual(level.original_image_path, str(replacement_path.resolve()))
        self.assertNotEqual(level.original_image_path, str(source_path))
        self.assertFalse(thread.output_path.exists())
        warning.assert_called_once()
        self.assertIn("was not applied", warning.call_args.args[2])
        failed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(failed_job.status, JOB_STATUS_FAILED)  # type: ignore[union-attr]

    def test_changed_plan_transform_rejects_stale_metric_result(self) -> None:
        self._load_source_image()
        thread, job_id = self._start_correction()
        thread.result = self._successful_result(
            thread,
            size=_plan_metric_calibration().normalized_image_size,
            metric_calibration=_plan_metric_calibration(),
        )
        level = self.workspace.current_level
        level.canvas_offset_x_pixels = 25.0

        with patch("housemaker.main.QMessageBox.warning") as warning:
            thread.finish()

        self.assertNotEqual(level.image_path, str(thread.output_path.resolve()))
        self.assertFalse(thread.output_path.exists())
        self.assertEqual(level.canvas_offset_x_pixels, 25.0)
        self.assertIsNone(level.plan_metric_calibration)
        warning.assert_called_once()
        self.assertIn("was not applied", warning.call_args.args[2])
        failed_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(failed_job.status, JOB_STATUS_FAILED)  # type: ignore[union-attr]

    def test_cancelled_correction_discards_output_after_worker_stops(self) -> None:
        self._load_source_image()
        thread, job_id = self._start_correction()
        thread.result = self._successful_result(thread)

        self.assertTrue(self.workspace.job_manager.cancel_job(job_id))
        self.assertTrue(thread.interruption_requested)
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Correcting...",
        )
        thread.was_cancelled = True
        thread.finish()

        self.assertFalse(thread.output_path.exists())
        self.assertNotIn(
            self.workspace.current_level.index,
            self.workspace._plan_image_correction_runtimes,
        )
        cancelled_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(
            cancelled_job.status,  # type: ignore[union-attr]
            JOB_STATUS_CANCELLED,
        )
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Image correction",
        )
        self.assertTrue(self.workspace.image_correction_button.isEnabled())

    def test_cancel_and_join_discards_partial_output_on_project_replacement(
        self,
    ) -> None:
        self._load_source_image()
        thread, job_id = self._start_correction()
        thread.output_path.parent.mkdir(parents=True, exist_ok=True)
        thread.output_path.write_bytes(b"partially written correction")
        self.assertTrue(thread.output_path.exists())

        self.workspace._cancel_and_join_plan_image_corrections()

        self.assertTrue(thread.interruption_requested)
        self.assertFalse(thread.running)
        self.assertTrue(thread.deleted_later)
        self.assertFalse(thread.output_path.exists())
        self.assertEqual(self.workspace._plan_image_correction_runtimes, {})
        cancelled_job = self.workspace.job_manager.get_job(job_id)
        self.assertEqual(
            cancelled_job.status,  # type: ignore[union-attr]
            JOB_STATUS_CANCELLED,
        )
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Image correction",
        )
        self.assertTrue(self.workspace.image_correction_button.isEnabled())

    def test_stale_finished_callback_does_not_remove_newer_runtime(self) -> None:
        source_path = self._load_source_image()
        level_index = self.workspace.current_level.index
        stale_thread = _FakeCorrectionThread(
            source_path,
            self.temporary_path / "stale.png",
            "stale-openai-key",
        )
        replacement_thread = _FakeCorrectionThread(
            source_path,
            self.temporary_path / "replacement.png",
            "replacement-openai-key",
        )
        replacement_thread.start()
        replacement_job = self.workspace.job_manager.create_job(
            kind="Image correction",
            requested_name="",
            default_name="Replacement correction",
            stage="Preparing plan image (0%)",
        )
        replacement_runtime = _PlanImageCorrectionRuntime(
            source_path=source_path,
            source_revision=_build_local_file_revision(source_path),
            output_path=replacement_thread.output_path,
            job_id=replacement_job.job_id,
            thread=replacement_thread,  # type: ignore[arg-type]
        )
        self.workspace._plan_image_correction_runtimes[level_index] = (
            replacement_runtime
        )

        self.workspace._handle_plan_image_correction_finished(
            level_index,
            "stale-job-id",
            stale_thread,  # type: ignore[arg-type]
        )

        self.assertIs(
            self.workspace._plan_image_correction_runtimes[level_index],
            replacement_runtime,
        )
        self.assertTrue(replacement_thread.running)
        self.assertEqual(
            self.workspace.image_correction_button.text(),
            "Correcting...",
        )
        self.assertFalse(self.workspace.image_correction_button.isEnabled())


# ### Direct execution ###
if __name__ == "__main__":
    unittest.main()
